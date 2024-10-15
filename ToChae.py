import os
import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler

# 设置 PyTorch 线程数和设备
torch.set_num_threads(8)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置本地模型路径
model_path = "stable-diffusion-inpainting"  # 替换为你的模型文件夹路径

# 加载预训练模型
ckpt_path = os.path.join(model_path, "sd-v1-5-inpainting.ckpt")
ckpt = torch.load(ckpt_path, map_location=device)

# 假设 ckpt 包含了模型的完整 state_dict
unet = UNet2DConditionModel.from_config(os.path.join(model_path, "unet", "config.json"))
load_result = unet.load_state_dict(ckpt['state_dict'], strict=False)  # 确认 'model' 是正确的键，根据实际情况可能需要调整

# 打印加载结果
print("Loaded keys:", len(ckpt['state_dict']) if 'state_dict' in ckpt else len(ckpt))  # 显示加载了多少个键
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)
print("Incorrect shapes:", getattr(load_result, 'incorrect_shapes', 'Not available'))  # 如果有形状不匹配的情况，这将被显示

# 加载其他必要的组件
text_encoder = CLIPTextModel.from_pretrained(os.path.join(model_path, "text_encoder"))
tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
scheduler = PNDMScheduler.from_pretrained(os.path.join(model_path, "scheduler"))
feature_extractor = CLIPImageProcessor.from_pretrained(os.path.join(model_path, "feature_extractor"))

# 创建 Stable Diffusion Inpaint 管道
pipe = StableDiffusionInpaintPipeline(
    vae=unet,  # 这里使用 unet 作为示例，你需要根据实际模型结构调整
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,  # 如果不需要安全检查器
    feature_extractor=feature_extractor
)
pipe.to(device)
