import torch
from diffusers import FluxPipeline
from PIL import Image

# 初始化模型和局部重绘类
model = FLUX()
inpainting = Inpainting(model)

# 加载模型参数
model.load_model('/mnt/sessd/ai_flux/FLUX.1-schnell')

# 原始图像和掩码图像路径
original_image_path = "/mnt/sessd/ai_tools/static/uploads/3fc3cd52ced292773a155b9e9f90933b.png"
mask_image_path = "/mnt/sessd/ai_tools/static/uploads/big_mask_3fc3cd52ced292773a155b9e9f90933b.png"

# 读取图像
original_image = Image.open(original_image_path)
mask_image = Image.open(mask_image_path)

# 设置参数
prompt = "A beautiful sunset over the mountains"
num_inference_steps = 50
guidance_scale = 7.5
seed = 42

# 重绘图像
repainted_image = model.inpaint(
    original_image,
    mask_image,
    prompt,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    seed=seed
)

# 保存或显示重绘后的图像
torch.save(repainted_image, "path_to_save_repainted_image.png")