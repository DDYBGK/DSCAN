import torch
import time
from thop import profile
from models.pointmlp import pointMLP  # 假设你的模型类在 pointmlp.py 中

# 1. 定义输入数据形状
batch_size = 1  # 批量大小
num_points = 1024  # 点云数量
input_shape = (batch_size, 3, num_points)  # 输入形状 (batch_size, channels, num_points)

# 2. 创建随机输入数据
input_tensor = torch.randn(input_shape)  # 随机生成输入数据

# 3. 实例化模型
model = pointMLP()  # 假设你的模型类为 pointMLP
model.eval()  # 设置为评估模式

# 4. 测试 FLOPs 和参数量
flops, params = profile(model, inputs=(input_tensor,))
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 打印 FLOPs（以 GFLOPs 为单位）
print(f"Parameters: {params / 1e6:.2f} M")  # 打印参数量（以百万为单位）

# 5. 测试推理时间
warmup_steps = 10  # 预热步骤，避免初始化的影响
test_steps = 100  # 测试步骤

# 预热
for _ in range(warmup_steps):
    _ = model(input_tensor)

# 正式测试
start_time = time.time()
for _ in range(test_steps):
    with torch.no_grad():  # 禁用梯度计算
        _ = model(input_tensor)
end_time = time.time()

# 计算平均推理时间
avg_inference_time = (end_time - start_time) / test_steps
print(f"Average Inference Time: {avg_inference_time * 1000:.2f} ms")  # 打印平均推理时间（以毫秒为单位）