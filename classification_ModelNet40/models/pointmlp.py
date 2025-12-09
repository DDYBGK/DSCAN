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
        x_aligned = torch.einsum('cd, bdn -> bcn', self.affine_matrix, x)
        out = self.mlp(x_aligned)
        return out

class GeometricStream(nn.Module):
    def __init__(self, activation='relu'):
        super(GeometricStream, self).__init__()
        self.stage1 = GeometricAffineModule(3, 64, activation=activation)
        self.stage2 = GeometricAffineModule(64, 128, activation=activation)
        self.stage3 = GeometricAffineModule(128, 256, activation=activation)
        self.stage4 = GeometricAffineModule(256, 512, activation=activation)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x 

# ====================================================================================
# =================== 核心模块 2: 交叉注意力 (CCSA 使用) ===================
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
# =========================== PointMLP 骨干组件 ============================
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
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
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
    def forward(self, x): return self.act(self.net2(self.net1(x)) + x)

class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu', use_xyz=True):
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation))
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
            operation.append(ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation))
        self.operation = nn.Sequential(*operation)
    def forward(self, x): return self.operation(x)

# ====================================================================================
# ============================ DSCAN 模型主体 ============================
# ====================================================================================

class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], 
                 pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2],
                 attn_d_model=256, attn_n_head=4,
                 geo_ratio=2, sem_ratio=3, fusion_dim=1024, # 默认值，会在调用时被覆盖
                 **kwargs):

        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = self.points

        # 1. 语义流编码器
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
                                             res_expansion=res_expansion, bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)

            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            self.channel_list.append(last_channel)

        # 2. 语义流 CCSA 模块
        self.fc_projection_list = nn.ModuleList()
        for i in range(self.stages):
            self.fc_projection_list.append(nn.Sequential(
                nn.Conv1d(self.channel_list[i], attn_d_model, 1, bias=False),
                nn.BatchNorm1d(attn_d_model),
                nn.ReLU(inplace=True)
            ))

        self.fp_module = PointNetFeaturePropagation(in_channel=attn_d_model, n_neighbors=3)
        self.cross_attn_1 = PointTransformerCrossAttention(d_model=attn_d_model, n_head=attn_n_head)
        self.cross_attn_2 = PointTransformerCrossAttention(d_model=attn_d_model, n_head=attn_n_head)
        self.act = get_activation(activation)

        # 3. 几何流
        self.geo_stream = GeometricStream(activation=activation)
        self.geo_output_dim = 512

        # 4. 融合投影层 (Ratio Fusion)
        total_ratio = geo_ratio + sem_ratio
        
        # 计算分配维度 (确保为整数)
        self.fusion_sem_dim = int(fusion_dim * (sem_ratio / total_ratio))
        self.fusion_geo_dim = fusion_dim - self.fusion_sem_dim
        
        self.sem_fusion_proj = nn.Sequential(
            nn.Linear(attn_d_model, self.fusion_sem_dim),
            nn.BatchNorm1d(self.fusion_sem_dim),
            get_activation(activation)
        )
        
        self.geo_fusion_proj = nn.Sequential(
            nn.Linear(self.geo_output_dim, self.fusion_geo_dim),
            nn.BatchNorm1d(self.fusion_geo_dim),
            get_activation(activation)
        )

        # 5. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num)
        )

    def forward(self, x):
        xyz_for_grouper = x.permute(0, 2, 1) # [B, N, 3]
        
        # ===== 1. 几何流 =====
        geo_feat = self.geo_stream(x)
        geo_global_feat = F.adaptive_max_pool1d(geo_feat, 1).squeeze(dim=-1)
        
        # ===== 2. 语义流 =====
        xyz = xyz_for_grouper
        x_sem = self.embedding(x)
        pos_features = []
        xyz_coords = []

        # Backbone
        for i in range(self.stages):
            xyz, x_sem = self.local_grouper_list[i](xyz, x_sem.permute(0, 2, 1))
            x_sem = self.pre_blocks_list[i](x_sem)
            x_sem = self.pos_blocks_list[i](x_sem)
            pos_features.append(x_sem)
            xyz_coords.append(xyz)

        # CCSA
        projected_features = [self.fc_projection_list[i](pos_features[i]).permute(0, 2, 1) for i in range(self.stages)]

        O_1_p, O_2_p, O_3_p, O_4_p = projected_features
        xyz_1, xyz_2, xyz_3, xyz_4 = xyz_coords

        V_interp = self.fp_module(xyz_q=xyz_3, xyz_k=xyz_4, v_k=O_4_p)
        V_new = self.cross_attn_1(q=O_2_p, k=O_3_p, v=V_interp, xyz_q=xyz_2, xyz_k=xyz_3)
        O1_fused = self.cross_attn_2(q=O_1_p, k=O_2_p, v=V_new, xyz_q=xyz_1, xyz_k=xyz_2)

        semantic_global_feat = F.adaptive_max_pool1d(O1_fused.permute(0, 2, 1), 1).squeeze(dim=-1)

        # ===== 3. 按比例融合 =====
        sem_proj = self.sem_fusion_proj(semantic_global_feat)
        geo_proj = self.geo_fusion_proj(geo_global_feat)
        final_feat = torch.cat([sem_proj, geo_proj], dim=1)

        x = self.classifier(final_feat)
        return x

def DSCAN_ModelNet40(num_classes=40, **kwargs) -> Model:
    """
    配置 DSCAN 模型以适应 ModelNet40 数据集
    参数修改依据论文:
    1. 层数配置: [1, 1, 2, 1] (与 ScanObjectNN 相同，同属分类任务 )
    2. 融合比例 (Geo/Sem): 1:1 (论文 4.1 节 ModelNet40 实验结果 )
    3. 类别数: 40 (ModelNet40 默认)
    """
    return Model(points=1024, 
                 class_num=num_classes, 
                 embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], 
                 # --- 保持与 ScanObjectNN 相同的层数配置 ---
                 pre_blocks=[1, 1, 2, 1], 
                 pos_blocks=[1, 1, 2, 1],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2],
                 attn_d_model=256, attn_n_head=4, 
                 # --- 修改点: 融合比例 1:1 ---
                 geo_ratio=1, sem_ratio=1, fusion_dim=1024,
                 **kwargs)

if __name__ == '__main__':
    # 模拟 ModelNet40 输入 (Batch=2, Channels=3, Points=1024)
    data = torch.rand(2, 3, 1024)
    print("===> testing DSCAN for ModelNet40 (Config: [1,1,2,1], Ratio 1:1) ...")
    
    # 初始化模型
    model = DSCAN_ModelNet40(num_classes=40)
    
    # 打印融合维度验证
    total_dim = model.fusion_geo_dim + model.fusion_sem_dim
    print(f"融合层配置:")
    print(f"  - 总维度: {total_dim}")
    print(f"  - 语义流 (Sem) 分配: {model.fusion_sem_dim} (50%)")
    print(f"  - 几何流 (Geo) 分配: {model.fusion_geo_dim} (50%)")
    
    out = model(data)
    print(f"输出形状: {out.shape} (预期: [2, 40])")