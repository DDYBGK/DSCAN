import torch
import torch.nn as nn
import torch.nn.functional as F


# from torch import einsum
# from einops import rearrange, repeat


# from pointnet2_ops import pointnet2_utils


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
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        fps_idx = farthest_point_sample(xyz, self.groups).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
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

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


# 创新  act((mlp[channels]->pool->mlp[channels*4]->mlp[256],bn)+x)
class InvResMLPs(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        # channel分别是 128 256 512 1024
        super(InvResMLPs, self).__init__()
        self.act = get_activation(activation)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel * 2,
                      kernel_size=kernel_size, groups=groups, bias=bias))
        self.mlp2 = nn.Sequential(
            nn.Conv1d(in_channels=channel * 2, out_channels=channel * 2,
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(channel * 2)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(in_channels=channel * 2, out_channels=channel,
                      kernel_size=kernel_size, groups=groups, bias=bias))

    def forward(self, x):
        return self.act(self.mlp3(self.mlp2(self.mlp1(x))) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                # ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                #                 bias=bias, activation=activation)
                InvResMLPs(out_channels, groups=groups, res_expansion=res_expansion,
                           bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointAgent(nn.Module):
    def __init__(self):
        super(PointAgent, self).__init__()
        self.fc_pre_agent_list = nn.ModuleList()
        last_channel = [128, 256, 512, 1024]

        for i in range(4):
            self.fc_pre_agent_list.append(nn.Sequential(
                nn.Conv1d(in_channels=last_channel[i], out_channels=1024, kernel_size=1, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            ))
        # 1. 代理token压缩模块（修改第一层卷积参数）
        self.agent_compression = nn.MaxPool1d(kernel_size=8, stride=8)  # [B, 1024, 256] -> [B, 1024, 32]

    def forward(self, qakv):
        """
        输入形状说明：qakv是一个张量列表
        qakv[0]: [batch, 128, 512]
        qakv[1]: [batch, 256, 256]
        qakv[2]: [batch, 512, 128]
        qakv[3]: [batch, 1024, 64]
        """
        pre_features = []
        for i in range(4):
            pre_feat = self.fc_pre_agent_list[i](qakv[i])
            pre_features.append(pre_feat)
        q = pre_features[0]  # q: [batch, 1024, 512]
        a = pre_features[1]  # a: [batch, 1024, 256]
        k = pre_features[2]  # k: [batch, 1024, 128]
        v = pre_features[3]  # v: [batch, 1024, 64]
        batch_size = q.size(0)

        # 1. 压缩代理token到32个点
        a_compressed = self.agent_compression(a)  # [B,1024,256] -> [B,1024,32]
        # print(f"a_compressed形状是{a_compressed.shape}")
        # 2. 值token扩展（通过concat实现）
        v_expanded = torch.cat([v, v], dim=-1)  # [B,1024,64] -> [B,1024,128]
        # print(v_expanded.shape)
        # 3. Agent聚合阶段 (A-K-V注意力)
        attn_weights = torch.matmul(a_compressed.transpose(1, 2), k) / (1024 ** 0.5)  # [B,32,128]
        attn_weights = F.softmax(attn_weights, dim=-1)
        v_new = torch.matmul(attn_weights, v_expanded.transpose(1, 2))  # [B,32,1024]
        v_new = v_new.transpose(1, 2)  # [B,1024,32]

        # 4. Agent广播阶段 (Q-A-V_new注意力)
        q = q.transpose(1, 2)  # [B,512,1024]
        attn_scores = torch.matmul(q, a_compressed) / (1024 ** 0.5)  # [B,512,32]
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v_new.transpose(1, 2))  # [B,512,1024]

        return output.transpose(1, 2)  # 恢复通道维度 [B,1024,512]


class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        self.point_agent = PointAgent()
        self.pre_mha_list = nn.ModuleList()
        self.pos_mha_list = nn.ModuleList()
        self.MHA = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True, bias=False)
        self.fc_pre_mha_list = nn.ModuleList()
        self.q_linear = nn.Sequential(
                nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            )
        self.k_linear = nn.Sequential(
                nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            )
        self.v_linear = nn.Sequential(
                nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            )
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            self.fc_pre_mha_list.append(nn.Sequential(
                nn.Conv1d(in_channels=out_channel, out_channels=1024, kernel_size=1, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            ))

            self.pre_mha_list.append(nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True, bias=False))
            self.pos_mha_list.append(nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True, bias=False))
            self.fc_pre_mha_list.append(nn.Sequential(
                nn.Conv1d(in_channels=out_channel, out_channels=1024, kernel_size=1, bias = False),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            ))
        # 多头注意力机制mha
        # self.mha = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True, bias=False)
        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
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
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        pos_features = []
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            print(x.shape)
            pos_features.append(x)
        x_org = x
        post_mha_feature_buffer = []
        for i in range(self.stages):
            pre_mha_feats = self.fc_pre_mha_list[i](pos_features[i])
            q = (pre_mha_feats).permute(0, 2, 1)  #2,1024,512 -> 2,512,1024
            k = (pre_mha_feats).permute(0, 2, 1)
            v = (pre_mha_feats).permute(0, 2, 1)
            post_mha_feats, _ = self.pre_mha_list[i](q, k, v)
            print(post_mha_feats.shape)  #2 512 1024
            post_mha_feats, _ = self.pos_mha_list[i](post_mha_feats, post_mha_feats, post_mha_feats)
            post_mha_feature_buffer.append(post_mha_feats.permute(0, 2, 1))
        x = torch.cat(post_mha_feature_buffer, dim=2)
        print(x.shape)
        x = F.adaptive_max_pool1d(x, output_size=512)
        print(f"x经512最大池化后{x.shape}")
        q = self.q_linear(x).permute(0, 2, 1);
        print(f"q形状{q.shape}")
        k = self.k_linear(x).permute(0, 2, 1);
        print(f"k形状{k.shape}")
        v = self.v_linear(x).permute(0, 2, 1);
        print(f"v形状{v.shape}")
        x_final, _ = self.MHA(q, k, v)
        x = F.adaptive_max_pool1d(x_final, 1).squeeze(dim=-1)
        # after_x = self.point_agent(pos_features) #B 1024 512
        # pooled_after_x = F.adaptive_max_pool1d(after_x, output_size=32)
        # pooled_x_org = F.adaptive_max_pool1d(x_org, output_size=32)
        #x_final = torch.cat((pooled_x_org, pooled_after_x), dim=-1)
        # print(f"x_final的形状是{x_final.shape}")
        # qakv = []
        # for i in range(self.stages):
        #     pre_feat = self.fc_pre_mha_list[i](pos_features[i])
        #     qakv.append(pre_feat)
        # pool = torch.nn.AdaptiveAvgPool1d(64)
        # q = qakv[0]
        # a = qakv[1]                     #2 1024 256
        # k = qakv[2]                     #2 1024 128
        # v = qakv[3]                     #2 1024 64
        # k_pooled = (pool(k)).permute(0, 2, 1)
        # v = v.permute(0, 2, 1)
        # a_asQ = (pool(a)).permute(0, 2, 1)  #48 1024 64
        # new_v, _ = self.mha(a_asQ, k_pooled, v)    #48 1024 64
        # q_pooled = (pool(q)).permute(0, 2, 1)  #48 1024 512 -> 48 1024 64
        # a_asK = a_asQ
        # mha_feat_global, _ = self.mha(q_pooled, a_asK, new_v)
        # mha_feat_global = mha_feat_global.permute(0, 2, 1) # b 1024 64
        # x_new = 0.5*x_org + 0.5*mha_feat_global
        #x = F.adaptive_max_pool1d(x_final, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


def pointMLP(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pointMLPElite(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=32, groups=1, res_expansion=0.25,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    print("===> testing pointMLP ...")
    model = pointMLP()
    out = model(data)
    print(out.shape)

