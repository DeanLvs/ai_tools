from PIL import Image
import os
import torch
import gc,json
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, ControlNetModel, UNet2DConditionModel, AutoencoderKL, PNDMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPTextModelWithProjection
from safetensors.torch import load_file as load_safetensors
from ccxx import convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint
from convert_original_stable_diffusion_to_diffusers import convert_ldm_vae_checkpoint, convert_diffusers_name_to_compvis
from convert_diffusers_to_sd import KeyMap

class CustomInpaintPipeline:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CustomInpaintPipeline, cls).__new__(cls)
        return cls._instance
    def __init__(self, model_path="/Users/dean/sd15/stable-diffusion-v1-5",
                 controlnet_list_t = ['key_points'], use_out_test = False):
        if not hasattr(self, "_initialized"):
            self.cache = {}  # 内存缓存字典
            self._initialized = True
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_path = model_path
            self.controlnet_list = controlnet_list_t #,'depth','canny','key_points','depth',,'seg'
            self.text_device = self.device
            # 手动加载第一个文本编码器
            text_encoder = CLIPTextModel.from_pretrained(os.path.join(self.model_path, "text_encoder")).to(self.text_device) # , torch_dtype=torch.float16
            # 手动加载第一个分词器
            tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer"))
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer
            self._print_component_load_info("Text Encoder 1", text_encoder)
            self.pipe = self._initialize_pipe()
            weights_path = "/Users/dean/sd15/uberRealisticPornMerge_v23Final.safetensors"
            self.load_model_weights(weights_path)
        print('suc lora')
    def _initialize_pipe(self):
        # 手动加载调度器
        # scheduler = PNDMScheduler.from_pretrained(os.path.join(self.model_path, "scheduler")) # ,  use_karras_sigmas=True
        scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"),
                                                                solver_order=2, algorithm_type="sde-dpmsolver++",
                                                                use_karras_sigmas=True)
        vae = AutoencoderKL.from_pretrained(os.path.join(self.model_path, "vae")).to(self.device) # , torch_dtype=torch.float16
        # 手动加载 UNet
        unet = UNet2DConditionModel.from_pretrained(os.path.join(self.model_path, "unet")).to(self.device) #, torch_dtype=torch.float16
        # # 将各组件的键写入文件
        # self._write_keys_to_file("vae_keys.txt", vae.state_dict().keys())
        # self._write_keys_to_file("unet_keys.txt", unet.state_dict().keys())
        self._print_component_load_info("UNet", unet)
        self._print_component_load_info("VAE", vae)
        print(f"total keys: {len(unet.state_dict().keys()) + len(vae.state_dict().keys())} "
              f"text_encoder keys:  {len(self.text_encoder.state_dict().keys())} ")
        controlnet_init_list = []
        for controlnet_item in self.controlnet_list:
            if 'key_points' == controlnet_item:
                controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet)
            elif 'depth' == controlnet_item:
                controlnet_dept = ControlNetModel.from_pretrained( "lllyasviel/sd-controlnet-depth",torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet_dept)
            elif 'seg' == controlnet_item:
                controlnet_seg = ControlNetModel.from_pretrained( "lllyasviel/sd-controlnet-seg",torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet_seg)
            elif 'canny' == controlnet_item:
                controlnet_canny = ControlNetModel.from_pretrained( "lllyasviel/sd-controlnet-scribble",torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet_canny)
        # 创建自定义管道
        if len(controlnet_init_list) == 0:
            to_controlnet = None
        elif len(controlnet_init_list) == 1:
            to_controlnet = controlnet_init_list[0]
        else:
            to_controlnet = controlnet_init_list
        pipe = StableDiffusionPipeline(
            # controlnet=to_controlnet,
            vae=vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=None,
            safety_checker=None
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

    def _transform_keys_with_keymap(self, merge_state_dict):
        transformed_dict = {}
        for k, v in merge_state_dict.items():
            if k in KeyMap:
                new_key = KeyMap[k]
                transformed_dict[new_key] = v
            else:
                transformed_dict[k] = v
        return transformed_dict
    def load_model_weights(self, checkpoint_path, lor_h=False):
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
        print(
            f"{checkpoint_path} Total Text Encoder keys: {total_text_encoder_keys}, Loaded Text Encoder keys: {loaded_text_encoder_keys}")

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


    def unload_lora_weights(self):
        print("Un Loading LoRA layers:")
        self.pipe.unload_lora_weights()
        print(f"Un loaded lora")

    def load_lora_weights(self, lora_path=None, adapter_name=None, lora_scale=0.7):
        print("Loaded LoRA layers:")
        pretrained_model_name_or_path_or_dict = lora_path
        if isinstance(lora_path, dict):
            pretrained_model_name_or_path_or_dict = lora_path.copy()

        print(f"loaded lora path: {lora_path}\n")

    def generate_image(self, prompt="a sexy milf",
                       reverse_prompt="",#deformed, bad anatomy, mutated, long neck, narrow Hips
                       num_inference_steps = 40, seed = 178, guidance_scale = 6, progress_callback=None,
                       strength=1.0, control_image=None, controlnet_conditioning_scale=None):
        torch.manual_seed(seed)
        result = self.pipe(
                prompt = prompt,
                negative_prompt= reverse_prompt,
                num_inference_steps=num_inference_steps,
                # guidance_scale=guidance_scale,
                callback=progress_callback,
                # strength=strength,
                # image=control_image,
                # controlnet_conditioning_scale=controlnet_conditioning_scale,
                # padding_mask_crop=32,
                callback_steps=1  # 调用回调的步数
            ).images[0]
        torch.cuda.empty_cache()
        gc.collect()
        return result
if __name__ == '__main__':
    pip = CustomInpaintPipeline(controlnet_list_t = [])
    # clear_org_i = Image.open("images_openpose.png").convert("RGB")
    # control_image = [clear_org_i]
    # controlnet_conditioning_scale = 1.0
    img = pip.generate_image(
        prompt="sexy woman , this woman is milf, big breasts, nip slip, gym, white short skirt, "
               "ass, showing pussy, clean realistic face, 4K, standing on the street", reverse_prompt = "")
    img.save('miffffbbxxbff.png')

