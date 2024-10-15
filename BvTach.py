import torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline
from safetensors.torch import load_file
import torch.nn.functional as nnf

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

# 保存和显示掩码（可选）
mask.save("/Users/dean/Desktop/mask.png")
mask.show()

# 加载 V5.1 Hyper-Inpaint(VAE) 模型
model_path = "/Users/dean/PycharmProjects/ai_tools/pythonProject/stable-diffusion-2-inpainting/vae.safetensors"  # 替换为你的模型文件路径
state_dict = load_file(model_path)

# 加载模型架构和配置文件
config_path = "/path/to/your/config.json"  # 替换为你的配置文件路径
model_arch_path = "/path/to/your/model_architecture_file.py"  # 替换为你的模型架构文件路径

# 动态加载模型架构
import importlib.util

spec = importlib.util.spec_from_file_location("StableDiffusionInpaintPipeline", model_arch_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# 创建 Stable Diffusion Inpaint 管道
pipe = model_module.StableDiffusionInpaintPipeline.from_pretrained(config_path, state_dict=state_dict, torch_dtype=torch.float16)

# 使用 GPU 如果可用
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# 生成图像
prompt = "A futuristic cityscape at sunset"
with torch.no_grad():
    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]

# 保存生成的图像
result.save("/Users/dean/Desktop/output.png")
result.show()