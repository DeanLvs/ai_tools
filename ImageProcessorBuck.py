import os
import torch
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, SegformerImageProcessor, SegformerForSemanticSegmentation, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
import torch.nn.functional as nnf
from safetensors.torch import load_file as load_safetensors
from convert_diffusers_to_sd import KeyMap
import json
from convert_original_stable_diffusion_to_diffusers import create_vae_diffusers_config, convert_ldm_vae_checkpoint
from mask_generator import MaskGenerator

class ImageProcessor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ImageProcessor, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, model_path="stable-diffusion-inpainting", checkpoint_merge_path="/mnt/sessd/civitai-downloader/pornmasterPro_v8-inpainting.safetensors"):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.model_path = model_path
        self.checkpoint_merge_path = checkpoint_merge_path
        self._initialize_pipe()

    def _initialize_pipe(self):
        # 加载各个组件
        text_encoder = CLIPTextModel.from_pretrained(os.path.join(self.model_path, "text_encoder"))
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer"))
        scheduler = PNDMScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))

        # 手动加载 VAE 模型
        vae_config_path = os.path.join(self.model_path, "vae", "config.json")
        vae_weights_path = os.path.join(self.model_path, "vae", "diffusion_pytorch_model.fp16.bin")
        vae = AutoencoderKL.from_config(vae_config_path)
        vae.load_state_dict(torch.load(vae_weights_path, map_location=self.device))

        # 手动加载 UNet 模型
        unet_config_path = os.path.join(self.model_path, "unet", "config.json")
        unet_weights_path = os.path.join(self.model_path, "unet", "diffusion_pytorch_model.fp16.bin")
        unet = UNet2DConditionModel.from_config(unet_config_path)

        # 创建自定义的 Stable Diffusion Inpaint 管道
        self.pipe = StableDiffusionInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,  # 不加载安全检查器
            feature_extractor=CLIPImageProcessor.from_pretrained(os.path.join(self.model_path, "feature_extractor"))
        )
        self.pipe.to(self.device)
        self.load_model_weights()

    def load_model_weights(self):
        # 确保此路径和文件名正确
        if not os.path.exists(self.checkpoint_merge_path):
            raise FileNotFoundError(f"File not found: {self.checkpoint_merge_path}")

        merge_state_dict = load_safetensors(self.checkpoint_merge_path)
        # 加载 UNet 权重
        unet_keys = self._transform_keys_with_keymap(merge_state_dict)
        unet_load_info = self.pipe.unet.load_state_dict(unet_keys, strict=False)
        total_unet_keys = len(self.pipe.unet.state_dict().keys())
        loaded_unet_keys = total_unet_keys - len(unet_load_info.missing_keys)
        print(f"Total UNet keys: {total_unet_keys}, Loaded UNet keys: {loaded_unet_keys}")
        print(f"unet_load_info.missing_keys: {unet_load_info.missing_keys}")

        # 加载文本编码器权重
        text_encoder_keys = {k.replace("cond_stage_model.transformer.", ""): v for k, v in merge_state_dict.items() if
                             k.startswith("cond_stage_model.transformer.")}
        text_encoder_load_info = self.pipe.text_encoder.load_state_dict(text_encoder_keys, strict=False)
        total_text_encoder_keys = len(self.pipe.text_encoder.state_dict().keys())
        loaded_text_encoder_keys = total_text_encoder_keys - len(text_encoder_load_info.missing_keys)
        print(f"Total Text Encoder keys: {total_text_encoder_keys}, Loaded Text Encoder keys: {loaded_text_encoder_keys}")

        # 加载 VAE 权重
        with open(os.path.join(self.model_path, 'vae/config.json'), 'r') as file:
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
        vae_load_info = self.pipe.vae.load_state_dict(converted_vae_checkpoint, strict=False)
        total_vae_keys = len(self.pipe.vae.state_dict().keys())
        loaded_vae_keys = total_vae_keys - len(vae_load_info.missing_keys)
        print(f"Total VAE keys: {total_vae_keys}, Loaded VAE keys: {loaded_vae_keys}")

    def _transform_keys_with_keymap(self, merge_state_dict):
        transformed_dict = {}
        for k, v in merge_state_dict.items():
            if k in KeyMap:
                new_key = KeyMap[k]
                transformed_dict[new_key] = v
            else:
                transformed_dict[k] = v
        return transformed_dict

    def generate_image(self, image_path, mask, prompt, reverse_prompt, num_inference_steps=100, guidance_scale=9):
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size  # 记录原始图像尺寸

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                reverse_prompt=reverse_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        # 调整生成图像尺寸为原始图像的尺寸
        result = result.resize((original_width, original_height), Image.BILINEAR)
        return result


# 示例使用
if __name__ == '__main__':
    processor = ImageProcessor()
    image_path = "e256463acf35cf523ce83998427dbef6.png"
    image = Image.open(image_path).convert("RGB")
    mask_generator = MaskGenerator()
    mask = mask_generator.generate_mask(image)
    result = processor.generate_image(image_path, mask, "sexy naked woman, beautiful body, artistic nude", "no clothes, no dress, no shirt, no pants, no covering")
    result.save("output_image.png")
    result.show()