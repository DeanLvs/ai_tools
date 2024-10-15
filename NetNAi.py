import torch
from PIL import Image, ImageFilter
import torch.nn.functional as nnf
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline
from safetensors.torch import load_file

# 使用 GPU 如果可用
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载图像分割模型
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
segmentation_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
segmentation_model.to(device)

# 加载图像
image_path = "cb45ac85254a115f83761faff0c4e150.png"  # 替换为您的本地图片路径
image = Image.open(image_path).convert("RGB")
original_width, original_height = image.size  # 记录原始图像尺寸
inputs = processor(images=image, return_tensors="pt").to(device)

# 获取分割结果
with torch.no_grad():
    outputs = segmentation_model(**inputs)
logits = outputs.logits.cpu()

# 上采样到输入图像大小
upsampled_logits = nnf.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]

# 创建掩码，找到衣服区域（上衣、裙子、裤子、连衣裙）
mask = Image.new("L", image.size, 0)
for y in range(pred_seg.shape[0]):
    for x in range(pred_seg.shape[1]):
        if pred_seg[y, x].item() in [4, 5, 6, 7]:  # 上衣、裙子、裤子、连衣裙
            mask.putpixel((x, y), 255)

# 对掩码进行模糊处理，使边界更平滑
mask = mask.filter(ImageFilter.GaussianBlur(5))

# 保存掩码用于查看
mask.save("mask.png")

# 加载 stable-diffusion-inpainting 模型
model_path = "/mnt/sessd/civitai-downloader/pornmasterPro_v8-inpainting.safetensors"  # 下载的 Checkpoint Merge 模型路径
model_state_dict = load_file(model_path)

# 使用 diffusers 加载模型
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
pipe.unet.load_state_dict(model_state_dict, strict=False)
pipe.to(device)

# 生成修复图像
prompt = "a clean version of the image without clothes"  # 根据需要调整提示词
output = pipe(prompt=prompt, image=image, mask_image=mask).images

# 调整生成图像尺寸为原始图像的尺寸
output_image = output[0].resize((original_width, original_height), Image.BILINEAR)

# 保存输出图像
output_image.save("output_image.png")
output_image.show()