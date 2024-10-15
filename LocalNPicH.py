import os
import torch
from PIL import Image, ImageFilter
from transformers import  AutoImageProcessor, Mask2FormerForUniversalSegmentation, SegformerImageProcessor, SegformerForSemanticSegmentation, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
import torch.nn.functional as nnf
from safetensors.torch import load_file as load_safetensors
from convert_diffusers_to_sd import KeyMap
import json
from convert_original_stable_diffusion_to_diffusers import create_vae_diffusers_config, convert_ldm_vae_checkpoint


def transform_keys_with_keymap(merge_state_dict):
    transformed_dict = {}
    for k, v in merge_state_dict.items():
        if k in KeyMap:
            new_key = KeyMap[k]
            transformed_dict[new_key] = v
        else:
            transformed_dict[k] = v
    return transformed_dict
def loadModel(pipe, checkpoint_merge_path = "/mnt/sessd/civitai-downloader/pornmasterPro_v8-inpainting.safetensors"):
    # 使用 GPU 如果可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 确保此路径和文件名正确
    if not os.path.exists(checkpoint_merge_path):
        raise FileNotFoundError(f"File not found: {checkpoint_merge_path}")

    merge_state_dict = load_safetensors(checkpoint_merge_path)
    # 加载 UNet 权重
    unet_keys = transform_keys_with_keymap(merge_state_dict)
    unet_load_info = pipe.unet.load_state_dict(unet_keys, strict=False)
    total_unet_keys = len(pipe.unet.state_dict().keys())
    loaded_unet_keys = total_unet_keys - len(unet_load_info.missing_keys)
    print(f"Total UNet keys: {total_unet_keys}, Loaded UNet keys: {loaded_unet_keys}")
    print(f"unet_load_info.missing_keys: {unet_load_info.missing_keys}")

    # 加载文本编码器权重
    text_encoder_keys = {k.replace("cond_stage_model.transformer.", ""): v for k, v in merge_state_dict.items() if
                         k.startswith("cond_stage_model.transformer.")}
    text_encoder_load_info = pipe.text_encoder.load_state_dict(text_encoder_keys, strict=False)
    total_text_encoder_keys = len(pipe.text_encoder.state_dict().keys())
    loaded_text_encoder_keys = total_text_encoder_keys - len(text_encoder_load_info.missing_keys)
    print(f"Total Text Encoder keys: {total_text_encoder_keys}, Loaded Text Encoder keys: {loaded_text_encoder_keys}")

    # 加载 VAE 权重
    with open('stable-diffusion-inpainting/vae/config.json', 'r') as file:
        dataVae = json.load(file)
    # 构造config字典
    configVae = dict(
        sample_size=dataVae["sample_size"],
        in_channels=dataVae["in_channels"],
        out_channels=dataVae["out_channels"],
        down_block_types=tuple(dataVae["down_block_types"]),
        up_block_types=tuple(dataVae["up_block_types"]),
        block_out_channels=tuple(dataVae["block_out_channels"]),
        latent_channels=dataVae["latent_channels"],
        layers_per_block=dataVae["layers_per_block"],
    )

    converted_vae_checkpoint = convert_ldm_vae_checkpoint(merge_state_dict, configVae)
    vae_load_info = pipe.vae.load_state_dict(converted_vae_checkpoint, strict=False)
    total_vae_keys = len(pipe.vae.state_dict().keys())
    loaded_vae_keys = total_vae_keys - len(vae_load_info.missing_keys)
    print(f"Total v8 VAE keys: {total_vae_keys}, Loaded VAE keys: {loaded_vae_keys}")
def mark_mask(image):
    # 使用 GPU 如果可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载图像分割模型
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    segmentation_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    segmentation_model.to(device)
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
    return mask


def mark_human_body_without_head(image_path):
    # 加载图像处理器和模型
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")

    # 加载图像并进行预处理
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # 获取分割结果
    prediction = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    segmentation = prediction['segmentation']
    segments_info = prediction['segments_info']

    # 获取标签信息
    person_segments = [segment for segment in segments_info if segment['category_id'] == 0]  # "person" 标签是0

    # 创建掩码
    mask = Image.new("L", image.size, 0)
    for segment in person_segments:
        segment_id = segment['id']
        for y in range(segmentation.shape[0]):
            for x in range(segmentation.shape[1]):
                if segmentation[y, x] == segment_id:
                    mask.putpixel((x, y), 255)

    # 对掩码进行模糊处理，使边界更平滑
    mask = mask.filter(ImageFilter.GaussianBlur(5))
    return mask
if __name__ == '__main__':

    # 设置 PyTorch 线程数x
    torch.set_num_threads(8)

    # 使用 GPU 如果可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载图像
    image = Image.open("cb45ac85254a115f83761faff0c4e150.png").convert("RGB")

    original_width, original_height = image.size  # 记录原始图像尺寸

    mask = mark_mask(image)
    # 设置本地模型路径
    model_path = "stable-diffusion-inpainting"  # 替换为你的模型文件夹路径
    # 加载各个组件
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(model_path, "text_encoder"))
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
    scheduler = PNDMScheduler.from_pretrained(os.path.join(model_path, "scheduler"))
    # 手动加载 VAE 模型
    vae_config_path = os.path.join(model_path, "vae", "config.json")
    vae_weights_path = os.path.join(model_path, "vae", "diffusion_pytorch_model.fp16.bin")
    vae = AutoencoderKL.from_config(vae_config_path)
    vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
    # 手动加载 UNet 模型
    unet_config_path = os.path.join(model_path, "unet", "config.json")
    unet_weights_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.fp16.bin")
    unet = UNet2DConditionModel.from_config(unet_config_path)
    # 创建自定义的 Stable Diffusion Inpaint 管道
    pipe = StableDiffusionInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,  # 不加载安全检查器
        feature_extractor=CLIPImageProcessor.from_pretrained(os.path.join(model_path, "feature_extractor"))
    )
    pipe.to(device)
    loadModel(pipe)
    # 加载 Checkpoint Merge 模型权重
    # 生成图像
    prompt = "sexy naked woman, beautiful body, artistic nude"  # 根据需要调整提示词
    reverse_prompt = "no clothes, no dress, no shirt, no pants, no covering"
    num_inference_steps = 100  # 增加生成步骤
    guidance_scale = 9  # 调整指导尺度

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            reverse_prompt=reverse_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

    # 调整生成图像尺寸为原始图像的尺寸
    result = result.resize((original_width, original_height), Image.BILINEAR)

    # 保存生成的图像
    result.save("output80121.png")
    result.show()