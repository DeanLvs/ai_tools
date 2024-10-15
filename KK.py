import torch

# 加载 .ckpt 文件
ckpt_path = "stable-diffusion-inpainting/sd-v1-5-inpainting.ckpt"  # 替换为实际的文件路径
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

# 打印 .ckpt 文件中包含的所有键
print("Keys in checkpoint:", ckpt.keys())

# 如果 'state_dict' 在文件中，打印出 state_dict 的键
if 'state_dict' in ckpt:
    print("Keys in 'state_dict':", ckpt['state_dict'].keys())
else:
    # 如果没有 'state_dict'，直接打印顶级键下的内容
    for key, value in ckpt.items():
        print(f"{key}: {type(value)}")  # 这将告诉你每个键的数据类型
