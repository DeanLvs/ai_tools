import os
import torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
import torch.nn.functional as nnf
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import load_file
# 设置 PyTorch 线程数
torch.set_num_threads(8)
# 加载图像
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
segmentation_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
# 使用 GPU 如果可用
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载图像
image_path = "cb45ac85254a115f83761faff0c4e150.png"  # 替换为您的本地图片路径
image = Image.open(image_path).convert("RGB")
original_width, original_height = image.size  # 记录原始图像尺寸
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
# mask.save("/Users/dean/Desktop/mask.png")
# mask.show()

# 设置本地模型路径
model_path = "stable-diffusion-inpainting"  # 替换为你的模型文件夹路径

# 加载 Stable Diffusion Inpainting 的组件
text_encoder = CLIPTextModel.from_pretrained(os.path.join(model_path, "text_encoder"))
tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
scheduler = PNDMScheduler.from_pretrained(os.path.join(model_path, "scheduler"))

# 手动加载 UNet 和 VAE
#unet_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.fp16.safetensors")
#unet_state_dict = load_file(unet_path)
#unet_config_path = os.path.join(model_path, "unet", "config.json")
#config = UNet2DConditionModel.load_config(unet_config_path)
#unet = UNet2DConditionModel.from_config(config)
#unet.load_state_dict(unet_state_dict)


unet_path = "pornmasterPro_v8-inpainting.safetensors"  # 替换为您下载的 Checkpoint Merge 模型路径
unet_state_dict = load_safetensors(unet_path)
unet_config_path = os.path.join(model_path, "unet", "config.json")
config = UNet2DConditionModel.load_config(unet_config_path)
unet = UNet2DConditionModel.from_config(config)
unet.load_state_dict(unet_state_dict)


# 加载 LoRA 权重并合并到 UNet
# lora_path = "/Users/dean/Downloads/Better_cleavage.safetensors"
# lora_state_dict = load_safetensors(lora_path)
# unet.load_state_dict(lora_state_dict, strict=False)

# 加载多个 LoRA 权重并合并到 UNet
lora_paths = [
        "milfpeaches_CyberRealistic_Pony_09.safetensors",
        "longbingbing-000006.safetensors",
        "Full_BodySuit_By_Stable_Yogi.safetensors",
        "danielseav3.safetensors",
        "cici2013.safetensors",
        "body-painting-v3.1.safetensors",
        "Better_cleavage.safetensors",
        "Anya01V40.safetensors",
        "ackneellengthteeshirtBD2.safetensors"
        #"0287 bdsm bodysuit_v1.safetensors",
        #"0092 bodysuit 2_v1.safetensors"
        ]

for lora_path in lora_paths:
    print(lora_path)
    lora_state_dict = load_safetensors(lora_path)
    unet.load_state_dict(lora_state_dict, strict=False)


vae_path = os.path.join(model_path, "vae", "diffusion_pytorch_model.fp16.bin")
vae_state_dict = torch.load(vae_path, map_location=device)

vae_config_path = os.path.join(model_path, "vae", "config.json")
vae = AutoencoderKL.from_config(vae_config_path)
vae.load_state_dict(vae_state_dict)
# 加载 V5.1 Hyper-Inpaint(VAE) 权重
hyper_vae_path = "realisticVisionV60B1_v51HyperInpaintVAE.safetensors"  # 替换为你的权重文件路径
hyper_vae_state_dict = load_safetensors(hyper_vae_path)
# 将权重加载到模型
vae.load_state_dict(hyper_vae_state_dict, strict=False)

# 加载安全检查器
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_path, subfolder="safety_checker")

# 加载 feature extractor
feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")
print(str(device))
# 创建自定义的 Stable Diffusion Inpaint 管道
# 创建管道
pipe = StableDiffusionInpaintPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,  # 不加载安全检查器
    feature_extractor=None
)
pipe.to(device)

# 生成图像
prompt = "nude, nude body, revealing the skin, milf"
reverse_prompt = ""
num_inference_steps = 75  # 增加生成步骤
guidance_scale = 9  # 调整指导尺度
with torch.no_grad():
    result = pipe(prompt=prompt,
                  reverse_prompt=reverse_prompt,
                  image=image,
                  mask_image=mask,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale
                  ).images[0]
# 调整生成图像尺寸为原始图像的尺寸
result = result.resize((original_width, original_height), Image.BILINEAR)
# 保存生成的图像
result.save("output333.png")
result.show()
