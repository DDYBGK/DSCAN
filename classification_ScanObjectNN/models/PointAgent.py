import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # 3. 值token扩展模块（通过concat实现）
        # 替换为 MLP
        self.v_expansion = nn.Sequential(
            nn.Linear(64, 64)  # 线性层，将64通道处理
        )

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
        q = pre_features[0]  #q: [batch, 1024, 512]
        a = pre_features[1]  #a: [batch, 1024, 256]
        k = pre_features[2]  #k: [batch, 1024, 128]
        v = pre_features[3]  #v: [batch, 1024, 64]
        batch_size = q.size(0)

        # 1. 压缩代理token到32个点
        a_compressed = self.agent_compression(a)  # [B,1024,256] -> [B,1024,32]
        print(a_compressed.shape)
        # 2. 值token扩展（通过concat实现）
        v_expanded = torch.cat([v, self.v_expansion(v.permute(0,2,1)).permute(0,2,1)], dim=-1)  # [B,1024,64] -> [B,1024,128]
        print(v_expanded.shape)
        # 3. Agent聚合阶段 (A-K-V注意力)
        attn_weights = torch.matmul(a_compressed.transpose(1,2), k) / (1024**0.5)  # [B,32,128]
        attn_weights = F.softmax(attn_weights, dim=-1)
        v_new = torch.matmul(attn_weights, v_expanded.transpose(1,2))  # [B,32,1024]
        v_new = v_new.transpose(1,2)  # [B,1024,32]
        
        # 4. Agent广播阶段 (Q-A-V_new注意力)
        q = q.transpose(1,2)  # [B,512,1024]
        attn_scores = torch.matmul(q, a_compressed) / (1024**0.5)  # [B,512,32]
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v_new.transpose(1,2))  # [B,512,1024]
        
        return output.transpose(1,2)  # 恢复通道维度 [B,1024,512]

# 测试用例
if __name__ == "__main__":
    # 输入数据（含batch维度）
    batch_size = 2
    qakv_list = []
    q = torch.randn(batch_size, 128, 512)
    a = torch.randn(batch_size, 256, 256)
    k = torch.randn(batch_size, 512, 128)
    v = torch.randn(batch_size, 1024, 64)
    qakv_list.append(q)
    qakv_list.append(a)
    qakv_list.append(k)
    qakv_list.append(v)

    
    # 初始化模块
    agent_attn = PointAgent()
    
    # 前向传播
    output = agent_attn(qakv_list)
    print(f"输入q形状: {q.shape}")
    print(f"压缩后的a形状: [B,1024,32]")
    print(f"扩展后的v形状: [B,1024,128]")
    print(f"最终输出形状: {output.shape}")  # 应输出 [2,1024,512]