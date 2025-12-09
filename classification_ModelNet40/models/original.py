
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import einsum
# from einops import rearrange, repeat


#from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils


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
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        fps_idx = farthest_point_sample(xyz, self.groups).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint] B个物体每个物体npoint个点索引
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]  根据索引查找到新点坐标（center point）
        new_points = index_points(points, fps_idx)  # [B, npoint, d]  根据索引查找到新点特征（center feature）

        idx = knn_point(self.kneighbors, xyz, new_xyz)  #通过knn算法找到中心点周围nsample(32)个点的索引
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]  npoint是中心点个数
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points  #b g 3   b g k d

#mlp bn act
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
        #f0->(conv+bn+act)->f1->((conv+bn+act+conv+bn)+f0)->f
        return self.act(self.net2(self.net1(x)) + x)


#创新  act((mlp[channels]->pool->mlp[channels*4]->mlp[256],bn)+x)
class InvResMLPs(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        #channel分别是 128 256 512 1024
        super(InvResMLPs, self).__init__()
        self.act = get_activation(activation)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel*4,
                      kernel_size=kernel_size, groups=groups, bias=bias))
        self.mlp2 = nn.Sequential(
            nn.Conv1d(in_channels=channel*4, out_channels=channel*4,
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(channel*4)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(in_channels=channel*4, out_channels=channel,
                      kernel_size=kernel_size, groups=groups, bias=bias))
 
    def forward(self, x):
        return self.act(self.mlp3(self.mlp2(self.mlp1(x)))+x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        #out_channels分别是128 256 512 1024
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                InvResMLPs(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)  #32 512 6 32
        x = x.reshape(-1, d, s)  #32*512 6 32
        x = self.transfer(x)
        #1024 128 24 24是一个小集合24个点
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
        self.fc_pre_mha_list = nn.ModuleList()
        last_channel = embed_dim  #2 3 1024 -> 2 64 1024
        anchor_points = self.points #1024个点
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]  #128->256->512->1024
            pre_block_num = pre_blocks[i]  #2 2 2 2
            pos_block_num = pos_blocks[i]  #2 2 2 2
            kneighbor = k_neighbors[i] #32 32 32 32
            reduce = reducers[i]  #2 2 2 2
            #第一次循环之后anchor_points变成512
            #第二次循环之后anchor_points变成256
            #第三次循环之后anchor_points变成128
            #第四次循环之后anchor_points变成64  点数在不断的减小从而实现grouper的作用
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            #fir:64->128 sec:128->256 thi:256->512 for:512->1024
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            #经pos处理后变为[2,128,512] [2,256,256] [2,512,128] [2,1024,64]
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            self.fc_pre_mha_list.append(nn.Sequential(
                nn.Conv1d(in_channels=out_channel, out_channels=1024, kernel_size=1, bias = False),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            ))
        #多头注意力机制mha
        self.mha = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True, bias=False)
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
            x = self.pos_blocks_list[i](x)  # [b,d,g] [2,128,512], [2,256,256], [2,512,128], [2,1024,64]
            pos_features.append(x)
        x_org = x 
        qakv = []
        for i in range(self.stages):
            pre_feat = self.fc_pre_mha_list[i](pos_features[i])
            qakv.append(pre_feat)
        pool = torch.nn.AdaptiveAvgPool1d(64)
        q = qakv[0]                     #2 1024 512
        a = qakv[1]                     #2 1024 256
        k = qakv[2]                     #2 1024 128
        v = qakv[3]                     #2 1024 64
        k_pooled = (pool(k)).permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        a_asQ = (pool(a)).permute(0, 2, 1)  #48 1024 64
        new_v, _ = self.mha(a_asQ, k_pooled, v)    #48 1024 64
        q_pooled = (pool(q)).permute(0, 2, 1)  #48 1024 512 -> 48 1024 64
        a_asK = a_asQ
        mha_feat_global, _ = self.mha(q_pooled, a_asK, new_v)  #b 1024 g
        mha_feat_global = mha_feat_global.permute(0, 2, 1)
        x_new = 0.5*x_org + 0.5*mha_feat_global
        x = F.adaptive_max_pool1d(x_new, 1).squeeze(dim=-1)
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
                   k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2], **kwargs)

if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    print("===> testing pointMLP ...")
    model = pointMLP()
    out = model(data)
    print(out.shape)

