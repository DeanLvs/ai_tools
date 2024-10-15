from diffusers import StableDiffusionXLInpaintPipeline
import torch
from PIL import Image

# 加载 SDXL Inpainting 模型
pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
    "/mnt/essd/civitai-downloader/st",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

# 加载自定义的 UNet 权重（可选）
# pipeline.unet.load_state_dict(torch.load("path_to_unet_weights", map_location="cpu"))
# 使用示例
init_image_path = "/mnt/essd/ai_tools/static/uploads/588cb120a2ac4104a231101b20831e8f.jpeg"  # 替换为实际的初始图像路径
mask_image_path = "/mnt/essd/ai_tools/static/uploads/big_mask_588cb120a2ac4104a231101b20831e8f.jpeg"  # 替换为实际的遮罩图像路径

# 加载图像
init_image = Image.open(init_image_path).convert("RGB")
mask_image = Image.open(mask_image_path).convert("L")
original_width, original_height = init_image.size
prompt = "nude,Really detailed skin,reddit,Match the original pose,Match the original image angle,Match the light of the original image,natural body proportions"
result_image = pipeline.generate_image(prompt, "deformed,bad anatomy, mutated,long neck,disconnected limbs", init_image,
                                       mask_image)

# 使用管道进行推理
result = pipeline(prompt="your prompt here", image=init_image, mask_image=mask_image).images[0]
result.save("ou99999tput.png")