from PIL import Image
import os
import torch
from diffusers import StableDiffusionXLInpaintPipeline, UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPTextModelWithProjection
from safetensors.torch import load_file as load_safetensors
from transUnet import get_new_key
from transVae import transform_line

def replace_keys_unet(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = get_new_key(key)
        if new_key:
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def replace_keys_vae(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = transform_line(key)
        if new_key:
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

class CustomInpaintPipeline:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CustomInpaintPipeline, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path="stable-diffusion-xl-1.0-inpainting-0.1"):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_path = model_path
            self.pipe = self._initialize_pipe()
            weights_path = "/mnt/sessd/civitai-downloader/nudify_xl.safetensors"  # 替换为权重文件的实际路径
            self.load_pretrained_weights(weights_path)

    def _initialize_pipe(self):
        # 手动加载调度器
        scheduler = EulerDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))

        # 手动加载第一个文本编码器
        text_encoder = CLIPTextModel.from_pretrained(os.path.join(self.model_path, "text_encoder")).to(self.device)

        # 手动加载第二个文本编码器
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(os.path.join(self.model_path, "text_encoder_2")).to(self.device)

        # 手动加载第一个分词器
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer"))

        # 手动加载第二个分词器
        tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer_2"))

        # 手动加载 VAE
        vae = AutoencoderKL.from_pretrained(os.path.join(self.model_path, "vae")).to(self.device)

        # 手动加载 UNet
        unet = UNet2DConditionModel.from_pretrained(os.path.join(self.model_path, "unet")).to(self.device)
        # # 将各组件的键写入文件
        self._write_keys_to_file("vae_keys.txt", vae.state_dict().keys())
        self._write_keys_to_file("text_encoder_keys.txt", text_encoder.state_dict().keys())
        self._write_keys_to_file("text_encoder_2_keys.txt", text_encoder_2.state_dict().keys())
        self._write_keys_to_file("unet_keys.txt", unet.state_dict().keys())

        # 设置半精度
        vae = vae.half()
        text_encoder = text_encoder.half()
        text_encoder_2 = text_encoder_2.half()
        unet = unet.half()

        # 打印各组件的加载信息
        self._print_component_load_info("UNet", unet)
        self._print_component_load_info("VAE", vae)
        self._print_component_load_info("Text Encoder 1", text_encoder)
        self._print_component_load_info("Text Encoder 2", text_encoder_2)
        print(f"total keys: {len(unet.state_dict().keys()) + len(vae.state_dict().keys())} "
              f"text_encoder keys:  {len(text_encoder.state_dict().keys())} "
              f"text_encoder_2 keys: {len(text_encoder_2.state_dict().keys())}"
              )
        # 创建自定义管道
        pipe = StableDiffusionXLInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler
        )
        pipe.to(self.device)
        return pipe

    def _write_keys_to_file(self, filename, keys):
        with open(filename, 'w') as f:
            for key in keys:
                f.write(f"{key}\n")
        return
    def _print_component_load_info(self, component_name, component):
        state_dict = component.state_dict()
        total_keys = len(state_dict.keys())
        loaded_keys = sum(1 for k, v in state_dict.items() if v is not None)
        missing_keys = total_keys - loaded_keys

        print(f"{component_name} total keys: {total_keys}, loaded keys: {loaded_keys}, missing keys: {missing_keys}")

    def load_pretrained_weights(self, weights_path):
        if not weights_path or weights_path == '':
            print(f"no check point config")
            return
        # 加载预训练权重
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"权重文件未找到: {weights_path}")

        state_dict = load_safetensors(weights_path)

        # 替换 key
        unet_state_dict = replace_keys_unet(state_dict)
        unet_model_dict = self.pipe.unet.state_dict()
        loaded_unet = {}
        # 仅加载形状匹配的键
        for key in unet_state_dict.keys():
            if key in unet_model_dict:
                if unet_state_dict[key].shape == unet_model_dict[key].shape:
                    loaded_unet[key] = unet_state_dict[key]

        print(f"this {weights_path} len is {len(state_dict.keys())}")
        # 打印 state_dict 的所有键
        self._write_keys_to_file("weights_keys.txt", state_dict.keys())
        # 加载 UNet 权重
        print("Loading UNet weights...")
        unet_keys = unet_model_dict.keys()
        unet_load_info = self.pipe.unet.load_state_dict(loaded_unet, strict=False)
        print(f"UNet total keys: {len(unet_keys)}, loaded keys: {len(loaded_unet)}, missing keys: {len(unet_load_info.missing_keys)}")
        print(f"Missing UNet keys: {unet_load_info.missing_keys}\n")

        self._write_keys_to_file("Missingunet.txt", unet_load_info.missing_keys)

        vae_state_dict = replace_keys_vae(state_dict)
        # 加载 VAE 权重
        print("Loading VAE weights...")
        vae_model_dict = self.pipe.vae.state_dict()  # 获取完整的 VAE 模型状态字典
        loaded_vae = {}

        # 仅加载形状匹配的键
        for key in vae_state_dict.keys():
            if key in vae_model_dict:
                if vae_state_dict[key].shape == vae_model_dict[key].shape:
                    loaded_vae[key] = vae_state_dict[key]

        vae_load_info = self.pipe.vae.load_state_dict(loaded_vae, strict=False)
        print(
            f"VAE total keys: {len(vae_model_dict.keys())}, loaded keys: {len(loaded_vae)}, missing keys: {len(vae_load_info.missing_keys)}")
        print(f"Missing VAE keys: {vae_load_info.missing_keys}\n")

        self._write_keys_to_file("MissinguVAE.txt", vae_load_info.missing_keys)

        # 加载文本编码器权重
        print("Loading text encoder weights...")
        # 移除键名中的 'conditioner.embedders.0.transformer.' 前缀
        modified_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('conditioner.embedders.0.transformer.'):
                # 删除前缀，保留后续部分
                new_key = k.replace('conditioner.embedders.0.transformer.', '')
                modified_state_dict[new_key] = v
            else:
                modified_state_dict[k] = v
        print("Loading text encoder weights...")
        text_encoder_keys = self.pipe.text_encoder.state_dict().keys()
        loaded_text_encoder = modified_state_dict
        text_encoder_load_info = self.pipe.text_encoder.load_state_dict(loaded_text_encoder, strict=False)

        print(f"Text encoder total keys: {len(text_encoder_keys)}, loaded keys: {len(loaded_text_encoder)}, missing keys: {len(text_encoder_load_info.missing_keys)}")
        print(f"Missing text encoder keys: {text_encoder_load_info.missing_keys}\n")

        # 加载文本编码器权重
        print("Loading text encoder 2 weights...")
        # 移除键名中的 'conditioner.embedders.0.transformer.' 前缀
        print("Loading text encoder 2 weights...")
        text_encoder_keys = self.pipe.text_encoder_2.state_dict().keys()
        loaded_text_encoder = {}

        text_2_model_dict = self.pipe.text_encoder_2.state_dict()
        # 仅加载形状匹配的键
        for key in modified_state_dict.keys():
            if key in text_2_model_dict:
                if modified_state_dict[key].shape == text_2_model_dict[key].shape:
                    loaded_text_encoder[key] = modified_state_dict[key]

        text_encoder_load_info = self.pipe.text_encoder_2.load_state_dict(loaded_text_encoder, strict=False)
        print(
            f"Text encoder 2 total keys: {len(text_encoder_keys)}, loaded keys: {len(loaded_text_encoder)}, missing keys: {len(text_encoder_load_info.missing_keys)}")
        print(f"Missing text encoder 2 keys: {text_encoder_load_info.missing_keys}\n")

        self._write_keys_to_file("Missinguencoder2.txt", text_encoder_load_info.missing_keys)

    def generate_image(self, prompt, reverse_prompt, init_image, mask_image):
        result = self.pipe(prompt=prompt, reverse_prompt=reverse_prompt, image=init_image, mask_image=mask_image).images[0]
        return result

if __name__ == '__main__':
    # 使用示例
    pipeline = CustomInpaintPipeline()

