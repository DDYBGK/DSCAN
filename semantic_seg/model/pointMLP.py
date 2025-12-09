import torch
import torch.nn as nn
import torch.nn.functional as F


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
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
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


def farthest_point_sample(xyz, npoint):
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


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()
        fps_idx = farthest_point_sample(xyz, self.groups).long()
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
            if self.normalize == "anchor":
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


class InvResMLPs(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(InvResMLPs, self).__init__()
        self.act = get_activation(activation)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel * 4,
                      kernel_size=kernel_size, groups=groups, bias=bias))
        self.mlp2 = nn.Sequential(
            nn.Conv1d(in_channels=channel * 4, out_channels=channel * 4,
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(channel * 4)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(in_channels=channel * 4, out_channels=channel,
                      kernel_size=kernel_size, groups=groups, bias=bias))

    def forward(self, x):
        return self.act(self.mlp3(self.mlp2(self.mlp1(x))) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                InvResMLPs(out_channels, groups=groups, res_expansion=res_expansion,
                           bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)

    def forward(self, xyz1, xyz2, points1, points2):
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points


class PointMLP(nn.Module):
    def __init__(self, num_classes=13, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[2, 2, 2, 2],
                 gmp_dim=64, cls_dim=64, cls_num=None, **kwargs):
        super(PointMLP, self).__init__()

        # 兼容性处理：如果传入了 cls_num，优先使用 cls_num
        self.class_num = cls_num if cls_num is not None else num_classes

        self.stages = len(pre_blocks)
        self.points = points

        self.embedding = ConvBNReLU1D(9, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        self.fc_pre_mha_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]

        ### Building Encoder #####
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
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)

            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            en_dims.append(last_channel)

            # 为 Attention 准备的 projection
            self.fc_pre_mha_list.append(nn.Sequential(
                nn.Conv1d(in_channels=out_channel, out_channels=1024, kernel_size=1, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            ))

        self.mha = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True, bias=False)

        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0, en_dims[0])
        assert len(en_dims) == len(de_dims) == len(de_blocks) + 1
        for i in range(len(en_dims) - 1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i] + en_dims[i + 1], de_dims[i + 1],
                                           blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation)
            )

        self.act = get_activation(activation)

        # 语义分割头
        # 修正输入维度：1024 (Global Attention Feature) + 128 (Last Decoder Feature) = 1152
        self.conv1_cls = nn.Conv1d(1024 + de_dims[-1], 128, 1)
        self.bn1_cls = nn.BatchNorm1d(128)
        self.drop1_sem = nn.Dropout(0.5)
        self.conv2_cls = nn.Conv1d(128, self.class_num, 1)

    def forward(self, x):
        # x: [B, 9, N] (XYZRGB...)
        # 提取 XYZ 用于 grouping
        xyz = x.permute(0, 2, 1)[:, :, :3]
        x = self.embedding(x)

        feature_buffer = []
        xyz_list = [xyz]
        x_list = [x]

        # Encoder
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
            xyz_list.append(xyz)
            x_list.append(x)
            feature_buffer.append(x)

        # Attention Mechanism
        qakv = []
        for i in range(self.stages):
            pre_feat = self.fc_pre_mha_list[i](feature_buffer[i])
            qakv.append(pre_feat)

        pool = torch.nn.AdaptiveAvgPool1d(8)  # 保留了你的 pooling 设置
        q = qakv[0]
        a = qakv[1]
        k = qakv[2]
        v = qakv[3]

        k_pooled = (pool(k)).permute(0, 2, 1)
        v_perm = v.permute(0, 2, 1)
        a_asQ = (pool(a)).permute(0, 2, 1)

        new_v, _ = self.mha(a_asQ, k_pooled, v_perm)

        q_pooled = (pool(q)).permute(0, 2, 1)
        a_asK = a_asQ

        mha_feat_local, _ = self.mha(q_pooled, a_asK, new_v)
        mha_feat_local = F.adaptive_max_pool1d(mha_feat_local.permute(0, 2, 1), 128)
        mha_feat_global = F.adaptive_max_pool1d(mha_feat_local, 1)  # [B, 1024, 1]

        # 修复：动态获取当前 N，而不是写死 2048
        current_N = x_list[0].shape[-1]
        mha_feat_global = mha_feat_global.repeat(1, 1, current_N)  # [B, 1024, N]

        # Decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i + 1], xyz_list[i], x_list[i + 1], x)

        # Concatenate Global Feature + Decoder Output
        x_cat = torch.cat((mha_feat_global, x), dim=1)  # [B, 1024+128, N]

        # Segmentation Head
        x = self.drop1_sem(F.relu(self.bn1_cls(self.conv1_cls(x_cat))))
        x = self.conv2_cls(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  # [B, N, Class]
        return x


def pointMLP(num_classes=13, **kwargs) -> PointMLP:
    return PointMLP(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                    activation="relu", bias=True, use_xyz=True, normalize="anchor",
                    dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                    k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                    de_dims=[512, 256, 128, 128], de_blocks=[4, 4, 4, 4],
                    gmp_dim=64, cls_dim=64, **kwargs)