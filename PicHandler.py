import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageDraw, ImageFilter
import requests
import matplotlib.pyplot as plt
import torch.nn as nn


# 加载图像分割模型
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
segmentation_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# 加载图像
image_path = "/Users/dean/Desktop/e256463acf35cf523ce83998427dbef6.png"  # 替换为您的本地图片路径
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# 获取分割结果
with torch.no_grad():
    outputs = segmentation_model(**inputs)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]

# 创建掩码，找到衣服区域（上衣、裙子、裤子、连衣裙）
mask = Image.new("L", image.size, 0)
mask_draw = ImageDraw.Draw(mask)
for y in range(pred_seg.shape[0]):
    for x in range(pred_seg.shape[1]):
        if pred_seg[y, x] in [4, 5, 6, 7]:  # 上衣、裙子、裤子、连衣裙
            mask.putpixel((x, y), 255)

# 对掩码进行模糊处理，使边界更平滑
mask = mask.filter(ImageFilter.GaussianBlur(5))

# 可视化分割结果和掩码（可选）
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(pred_seg)
plt.title("Clothes Segmentation")
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.show()

# 加载Stable Diffusion模型
model_id = "runwayml/stable-diffusion-inpainting"
model_id = "Uminosachi/realisticVisionV51_v51VAE-inpainting"
model_id = "/Users/dean/Downloads/Better_cleavage.safetensors"
pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=torch.float32)

# 设置文本提示，使用更中性的描述
prompt = "remove clothes to let the skin leak out, full breasts"

# 进行图像修复，设置不同的种子
result = pipe(prompt=prompt, image=image, mask_image=mask, num_inference_steps=10, guidance_scale=10, generator=torch.manual_seed(42)).images[0]

# 保存结果
result.save("output_imag18.png")