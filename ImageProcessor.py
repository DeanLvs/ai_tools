import os
import torch
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler, DDIMScheduler
from safetensors.torch import load_file as load_safetensors
from convert_diffusers_to_sd import KeyMap
import json
from convert_original_stable_diffusion_to_diffusers import convert_ldm_vae_checkpoint, convert_diffusers_name_to_compvis
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
class ImageProcessor(metaclass=SingletonMeta):
    def __init__(self, model_path="stable-diffusion-inpainting", checkpoint_paths=None,sampler="DPM++ SDE Karras"):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.sampler = sampler
        self.checkpoint_paths = checkpoint_paths if checkpoint_paths else []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.model_path = model_path
        self._initialize_pipe()
        # 加载传入的模型权重
        for idx, checkpoint_path in enumerate(self.checkpoint_paths):
            try:
                if idx == 0:
                    self.load_model_weights(checkpoint_path)
                else:
                    self.load_model_weights(checkpoint_path, True)
            except Exception as ex:
                print(ex)
                continue
    def _initialize_pipe(self):
        # 手动加载调度器
        if self.sampler == "DPM++ SDE Karras":
            scheduler = PNDMScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        else:
            scheduler = DDIMScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))

        # 加载各个组件
        text_encoder = CLIPTextModel.from_pretrained(os.path.join(self.model_path, "text_encoder")).to(self.device)
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer"))

        # 手动加载 VAE 模型
        vae_config_path = os.path.join(self.model_path, "vae", "config.json")
        vae_weights_path = os.path.join(self.model_path, "vae", "diffusion_pytorch_model.fp16.bin")
        vae = AutoencoderKL.from_config(vae_config_path)
        vae.load_state_dict(torch.load(vae_weights_path, map_location=self.device))

        # 手动加载 UNet 模型
        unet_config_path = os.path.join(self.model_path, "unet", "config.json")
        unet_weights_path = os.path.join(self.model_path, "unet", "diffusion_pytorch_model.fp16.bin")
        unet = UNet2DConditionModel.from_config(unet_config_path).to(self.device)

        # 打印各组件的加载信息
        self._print_component_load_info("UNet", unet)
        self._print_component_load_info("VAE", vae)
        self._print_component_load_info("Text Encoder", text_encoder)

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
        # 打印各组件的加载信息
        self._print_component_load_info("UNet", unet)
        self._print_component_load_info("VAE", vae)
        self._print_component_load_info("Text Encoder", text_encoder)

    def load_model_weights(self, checkpoint_path, lor_h = False):
        # 确保此路径和文件名正确
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"File not found: {checkpoint_path}")
        merge_state_dict = load_safetensors(checkpoint_path)
        # 加载 UNet 权重

        if lor_h:
            transformed_dict_temp = {}
            for k, v in merge_state_dict.items():
                new_k = convert_diffusers_name_to_compvis(k, False)
                transformed_dict_temp[new_k] = v
            print(f"handler lor_h: {checkpoint_path}")
            merge_state_dict = transformed_dict_temp
        unet_keys = self._transform_keys_with_keymap(merge_state_dict)
        unet_load_info = self.pipe.unet.load_state_dict(unet_keys, strict=False)
        total_unet_keys = len(self.pipe.unet.state_dict().keys())
        loaded_unet_keys = total_unet_keys - len(unet_load_info.missing_keys)
        print(f"{checkpoint_path} Total UNet keys: {total_unet_keys}, Loaded UNet keys: {loaded_unet_keys}")
        print(f"{checkpoint_path} unet_load_info.missing_keys: {unet_load_info.missing_keys}")

        # 加载文本编码器权重
        text_encoder_keys = {k.replace("cond_stage_model.transformer.", ""): v for k, v in merge_state_dict.items() if
                             k.startswith("cond_stage_model.transformer.")}
        text_encoder_load_info = self.pipe.text_encoder.load_state_dict(text_encoder_keys, strict=False)
        total_text_encoder_keys = len(self.pipe.text_encoder.state_dict().keys())
        loaded_text_encoder_keys = total_text_encoder_keys - len(text_encoder_load_info.missing_keys)
        print(f"{checkpoint_path} Total Text Encoder keys: {total_text_encoder_keys}, Loaded Text Encoder keys: {loaded_text_encoder_keys}")

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
        print(f"{checkpoint_path} Total VAE keys: {total_vae_keys}, Loaded VAE keys: {loaded_vae_keys}")
        print(f"{checkpoint_path} vae_load_info.missing_keys: {vae_load_info.missing_keys}")

    def _print_component_load_info(self, component_name, component):
        state_dict = component.state_dict()
        total_keys = len(state_dict.keys())
        loaded_keys = sum(1 for k, v in state_dict.items() if v is not None)
        missing_keys = total_keys - loaded_keys
        print(f"{component_name} total keys: {total_keys}, loaded keys: {loaded_keys}, missing keys: {missing_keys}")
    def _transform_keys_with_keymap(self, merge_state_dict):
        transformed_dict = {}
        for k, v in merge_state_dict.items():
            if k in KeyMap:
                new_key = KeyMap[k]
                transformed_dict[new_key] = v
            else:
                transformed_dict[k] = v
        return transformed_dict
    def generate_image(self, prompt, reverse_prompt, init_image, mask_image):
        result = self.pipe(prompt=prompt, reverse_prompt=reverse_prompt, image=init_image, mask_image=mask_image).images[0]
        return result
if __name__ == '__main__':
    # 使用示例
    pipeline = ImageProcessor()
    weights_path = "/mnt/sessd/civitai-downloader/pornmasterPro_v8-inpainting-b.safetensors"  # 替换为权重文件的实际路径
    pipeline.load_model_weights(weights_path)

    # 使用示例
    init_image_path = "/mnt/sessd/ai_tools/static/uploads/588cb120a2ac4104a231101b20831e8f.jpeg"  # 替换为实际的初始图像路径
    mask_image_path = "/mnt/sessd/ai_tools/static/uploads/big_mask_588cb120a2ac4104a231101b20831e8f.jpeg"  # 替换为实际的遮罩图像路径

    # 加载图像
    init_image = Image.open(init_image_path).convert("RGB")
    mask_image = Image.open(mask_image_path).convert("L")

    prompt = "R/bigasses,R/pussy,R/boobs,nude,Really detailed skin,reddit,Match the original pose,Match the original image angle,Match the light of the original image,natural body proportions"
    result_image = pipeline.generate_image(prompt, "deformed,bad anatomy, mutated,long neck,disconnected limbs", init_image, mask_image)
    result_image.save("output123212.png")
