import torch

def print_checkpoint_keys(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # 检查是否是包含state_dict的字典
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print(f"Total keys in checkpoint '{checkpoint_path}': {len(state_dict.keys())}")
    for key in state_dict.keys():
        print(key)

if __name__ == "__main__":
    checkpoint_path = "pornmasterPro_v8-inpainting-b.safetensors"  # 请将此处替换为你的检查点文件路径
    print_checkpoint_keys(checkpoint_path)
