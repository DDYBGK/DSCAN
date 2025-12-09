import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils


# ====================================================================================
# ============================ 基础辅助函数 (保持不变) ============================
# ====================================================================================

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


# ====================================================================================
# ======================== 核心模块 1: 几何保持模块 (GP Module) ========================
# ====================================================================================

class GeometricAffineModule(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(GeometricAffineModule, self).__init__()
        self.affine_matrix = nn.Parameter(torch.eye(in_channels))
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            get_activation(activation)
        )

    def forward(self, x):
        # [B, C, N]
        x_aligned = torch.einsum('cd, bdn -> bcn', self.affine_matrix, x)
        out = self.mlp(x_aligned)
        return out


class GeometricStream(nn.Module):
    """
    几何流: 在分割任务中，该流直接处理原始N个点，不进行下采样，
    以保持完整的几何细节 (N x 512)。
    """

    def __init__(self, activation='relu'):
        super(GeometricStream, self).__init__()
        self.stage1 = GeometricAffineModule(3, 64, activation=activation)
        self.stage2 = GeometricAffineModule(64, 128, activation=activation)
        self.stage3 = GeometricAffineModule(128, 256, activation=activation)
        self.stage4 = GeometricAffineModule(256, 512, activation=activation)

    def forward(self, x):
        # x: [B, 3, N]
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x  # Output: [B, 512, N] (Segmentation keeps N)


# ====================================================================================
# =================== 核心模块 2: 交叉注意力 & FP (保持不变) ===================
# ====================================================================================

class PointTransformerCrossAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(PointTransformerCrossAttention, self).__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_head)
        )

    def forward(self, q, k, v, xyz_q, xyz_k):
        B, N_q, _ = q.shape
        N_k = k.shape[1]
        q_h = self.to_q(q).view(B, N_q, self.n_head, self.d_head).transpose(1, 2)
        k_h = self.to_k(k).view(B, N_k, self.n_head, self.d_head).transpose(1, 2)
        v_h = self.to_v(v).view(B, N_k, self.n_head, self.d_head).transpose(1, 2)

        xyz_q_r = xyz_q.unsqueeze(2).expand(-1, -1, N_k, -1)
        xyz_k_r = xyz_k.unsqueeze(1).expand(-1, N_q, -1, -1)
        rel_pos = xyz_q_r - xyz_k_r

        rpe = self.pos_mlp(rel_pos).permute(0, 3, 1, 2)
        energy = torch.einsum('bhid,bhjd->bhij', q_h, k_h) / (self.d_head ** 0.5)
        attn_energy = energy + rpe
        attn = F.softmax(attn_energy, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v_h)
        out = out.transpose(1, 2).contiguous().view(B, N_q, -1)
        return self.to_out(out)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp_channels=None, n_neighbors=3):
        super(PointNetFeaturePropagation, self).__init__()
        self.n_neighbors = n_neighbors

    def forward(self, xyz_q, xyz_k, v_k):
        B, N_q, _ = xyz_q.shape
        idx = knn_point(self.n_neighbors, xyz_k, xyz_q)
        sqrdists = square_distance(xyz_q, xyz_k)
        dist_nearest = torch.gather(sqrdists, 2, idx)
        dist_nearest = torch.clamp(dist_nearest, min=1e-10)
        weights = 1.0 / dist_nearest
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        grouped_v = index_points(v_k, idx)
        v_q = torch.einsum('bnk,bnkc->bnc', weights, grouped_v)
        return v_q


# ====================================================================================
# =========================== PointMLP 骨干组件 (保持不变) ============================
# ====================================================================================

class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize else None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            elif self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x): return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, int(channel * res_expansion), kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(int(channel * res_expansion), channel, kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(channel, channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(int(channel * res_expansion), channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu',
                 use_xyz=True):
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion, bias=bias,
                                             activation=activation))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2).reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation))
        self.operation = nn.Sequential(*operation)

    def forward(self, x): return self.operation(x)


# ====================================================================================
# =================== DSCAN 分割网络 (Segmentation Model) ===================
# ====================================================================================

class Model(nn.Module):
    def __init__(self, points=2048, class_num=50, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2],
                 # 默认参数 (会被覆盖)
                 pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2],
                 attn_d_model=256, attn_n_head=4,
                 # 融合参数
                 geo_ratio=1, sem_ratio=3, fusion_dim=128,
                 **kwargs):

        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.act = get_activation(activation)

        # ================== 1. 语义流 (Semantic Stream) ==================

        # --- Encoder (Backbone) ---
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = self.points

        self.channel_list = []
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce

            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)
            self.local_grouper_list.append(local_grouper)

            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation,
                                             use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)

            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            self.channel_list.append(last_channel)

        # --- Decoder (Feature Propagation for Segmentation) ---
        # 逐层上采样，恢复特征到 [B, N, C]
        self.decode_list = nn.ModuleList()

        # FP 4: Stage 4 -> Stage 3
        self.decode_list.append(PointNetFeaturePropagation(in_channel=self.channel_list[-1] + self.channel_list[-2],
                                                           mlp_channels=self.channel_list[-2]))
        # FP 3: Stage 3 -> Stage 2
        self.decode_list.append(PointNetFeaturePropagation(in_channel=self.channel_list[-2] + self.channel_list[-3],
                                                           mlp_channels=self.channel_list[-3]))
        # FP 2: Stage 2 -> Stage 1
        self.decode_list.append(PointNetFeaturePropagation(in_channel=self.channel_list[-3] + self.channel_list[-4],
                                                           mlp_channels=self.channel_list[-4]))
        # FP 1: Stage 1 -> Original (Input)
        self.decode_list.append(
            PointNetFeaturePropagation(in_channel=self.channel_list[-4] + embed_dim, mlp_channels=embed_dim))

        # ================== 2. 几何流 (Geometric Stream) ==================
        # 保持原始点数 N，输出 512 维特征
        self.geo_stream = GeometricStream(activation=activation)
        self.geo_output_dim = 512

        # ================== 3. 融合层 (Fusion) ==================
        total_ratio = geo_ratio + sem_ratio

        # 分配融合维度 (通常分割头维度较小，如 128)
        self.fusion_sem_dim = int(fusion_dim * (sem_ratio / total_ratio))
        self.fusion_geo_dim = fusion_dim - self.fusion_sem_dim

        # 语义特征投影 (FP1输出是 embed_dim, e.g., 64 -> fusion_sem_dim)
        self.sem_fusion_proj = nn.Sequential(
            nn.Conv1d(embed_dim, self.fusion_sem_dim, 1, bias=False),  # 从上采样后的维度映射
            nn.BatchNorm1d(self.fusion_sem_dim),
            self.act
        )

        # 几何特征投影 (512 -> fusion_geo_dim)
        self.geo_fusion_proj = nn.Sequential(
            nn.Conv1d(self.geo_output_dim, self.fusion_geo_dim, 1, bias=False),
            nn.BatchNorm1d(self.fusion_geo_dim),
            self.act
        )

        # ================== 4. 分割头 (Segmentation Head) ==================
        # 输入: fusion_dim + class_one_hot (通常ShapeNetPart会拼接类别向量，这里简化为直接预测)
        # 如果需要 label input，通常 concat label 到 feature map
        self.seg_head = nn.Sequential(
            nn.Conv1d(fusion_dim, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            self.act,
            nn.Dropout(0.5),
            nn.Conv1d(128, class_num, 1)
        )

    def forward(self, x):
        # x: [B, 3, N]
        B, C, N = x.shape
        xyz_original = x.permute(0, 2, 1)  # [B, N, 3]

        # ===== 1. 几何流 (Parallel) =====
        # [B, 512, N] (Full Resolution)
        geo_feat_full = self.geo_stream(x)

        # ===== 2. 语义流 (Encoder) =====
        xyz = xyz_original
        x_sem = self.embedding(x)  # [B, 64, N] (Level 0 features)

        l0_xyz = xyz
        l0_points = x_sem

        xyz_list = [l0_xyz]
        points_list = [l0_points]

        # Encoder Forward
        for i in range(self.stages):
            xyz, x_sem = self.local_grouper_list[i](xyz, x_sem.permute(0, 2, 1))
            x_sem = self.pre_blocks_list[i](x_sem)
            x_sem = self.pos_blocks_list[i](x_sem)
            xyz_list.append(xyz)  # [l0, l1, l2, l3, l4]
            points_list.append(x_sem)  # [f0, f1, f2, f3, f4]

        # ===== 3. 语义流 (Decoder / Upsampling) =====
        # 将特征从 Level 4 逐层上采样回 Level 0
        l4_points = points_list[4]
        l3_points = points_list[3]
        l2_points = points_list[2]
        l1_points = points_list[1]
        l0_points = points_list[0]

        l4_xyz = xyz_list[4]
        l3_xyz = xyz_list[3]
        l2_xyz = xyz_list[2]
        l1_xyz = xyz_list[1]
        l0_xyz = xyz_list[0]

        # FP4: l4 -> l3 (cat l3)
        # 注意: PointNetFeaturePropagation(xyz_q, xyz_k, v_k) -> interpolate v_k to xyz_q
        # 这里实际上我们需要 concat skip connection，简单的 FP 模块通常包含 concat 和 MLP
        # 由于上面定义的 FP 模块只做了 interpolation (weighted average)，我们需要手动 concat + projection?
        # 修正: PointNet++ 的 FP 模块通常自带 MLP。
        # 但上方提供的 PointNetFeaturePropagation 仅仅是 Interpolation。
        # 为了适应 PointMLP 常见结构，我们直接使用 Interpolation 结果与 Skip Connection 拼接，
        # 但通常需要一个 MLP 来处理拼接后的特征。
        # 这里的实现为了保持提供的代码一致性，我们假设 Decode 过程只做插值，
        # 或者我们需要修改 FP 模块。
        # 为了简单且有效，这里使用插值后直接与上一层特征拼接，作为下一层的输入。

        # Decode 4
        dist, idx = pointnet2_utils.three_nn(l3_xyz, l4_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_l4 = pointnet2_utils.three_interpolate(l4_points, idx, weight)
        l3_fused = torch.cat([l3_points, interpolated_l4], dim=1)  # [B, C3+C4, N3]

        # Decode 3
        dist, idx = pointnet2_utils.three_nn(l2_xyz, l3_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_l3 = pointnet2_utils.three_interpolate(l3_fused, idx, weight)
        l2_fused = torch.cat([l2_points, interpolated_l3], dim=1)

        # Decode 2
        dist, idx = pointnet2_utils.three_nn(l1_xyz, l2_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_l2 = pointnet2_utils.three_interpolate(l2_fused, idx, weight)
        l1_fused = torch.cat([l1_points, interpolated_l2], dim=1)

        # Decode 1 (Back to Original N=2048)
        dist, idx = pointnet2_utils.three_nn(l0_xyz, l1_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_l1 = pointnet2_utils.three_interpolate(l1_fused, idx, weight)
        sem_feat_full = torch.cat([l0_points, interpolated_l1], dim=1)  # [B, C_out, N]

        # ===== 4. 最终融合 (Final Fusion) =====
        # Projection
        # Sem: [B, C_total, N] -> [B, fusion_sem_dim, N]
        # Geo: [B, 512, N] -> [B, fusion_geo_dim, N]
        sem_proj = self.sem_fusion_proj(sem_feat_full)
        geo_proj = self.geo_fusion_proj(geo_feat_full)

        # Concat [B, fusion_dim, N]
        final_feat = torch.cat([sem_proj, geo_proj], dim=1)

        # ===== 5. 分割头预测 =====
        # [B, num_classes, N]
        x = self.seg_head(final_feat)

        return x


def DSCAN_ShapeNetPart(num_classes=50, **kwargs) -> Model:
    """
    配置 DSCAN 模型以适应 ShapeNet-Part 分割任务
    参数修改依据论文:
    1. 层数配置: [2, 2, 2, 2] (Page 8, Source 305)
    2. 融合比例 (Geo/Sem): 1:3 (Page 10, Source 349)
    3. 类别数: 50 (ShapeNet-Part parts)
    4. 输入点数: 2048
    """
    return Model(points=2048,
                 class_num=num_classes,
                 embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2],
                 # --- 修改点 1: 分割任务使用更深的网络 ---
                 pre_blocks=[2, 2, 2, 2],
                 pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2],
                 attn_d_model=256, attn_n_head=4,
                 # --- 修改点 2: 融合比例 1:3 ---
                 geo_ratio=1, sem_ratio=3, fusion_dim=128,
                 **kwargs)


if __name__ == '__main__':
    # 模拟 ShapeNet-Part 输入 (Batch=2, Channels=3, Points=2048)
    data = torch.rand(2, 3, 2048)
    print("===> testing DSCAN for ShapeNet-Part Segmentation...")
    print("     Config: Layers=[2,2,2,2], Ratio=1:3, Points=2048")

    # 初始化模型
    model = DSCAN_ShapeNetPart(num_classes=50)

    # 打印融合维度验证
    total_dim = model.fusion_geo_dim + model.fusion_sem_dim
    print(f"融合层配置:")
    print(f"  - 总维度 (Per-Point): {total_dim}")
    print(f"  - 语义流 (Sem) 分配: {model.fusion_sem_dim} (75%)")
    print(f"  - 几何流 (Geo) 分配: {model.fusion_geo_dim} (25%)")

    out = model(data)
    print(f"输出形状: {out.shape} (预期: [2, 50, 2048])")