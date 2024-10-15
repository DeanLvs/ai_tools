import os
from safetensors.torch import load_file as load_safetensors

def print_safetensors_keys(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 加载 safetensors 文件
    state_dict = load_safetensors(file_path)

    # 输出所有的 key
    for key in state_dict.keys():
        print(key)

if __name__ == "__main__":
    # 请将这里的路径替换为你的 safetensors 文件路径
    file_path = "/mnt/sessd/civitai-downloader/cleftofvenusV1.safetensors"
    print_safetensors_keys(file_path)