from diffusers import StableDiffusionPipeline
import torch

model_path = '/mnt/sessd/civitai-downloader/sdxxxl_v30.safetensors'

# 加载 SDXL 模型
pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant=None
)
# 替换权重
pipeline.load_checkpoint(model_path, torch_dtype=torch.float16,map_location="cuda")

# 使用模型
pipeline.to("cuda")

prompt = "sexy girl"
result = pipeline(prompt).images[0]
result.save("output.png")