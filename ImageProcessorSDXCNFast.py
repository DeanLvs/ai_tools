from PIL import Image
import os
import hashlib
import torch
import gc
from CustCoun import encode_prompt_with_cache
import SingletonOpenposeDetector
from diffusers import StableDiffusionXLControlNetInpaintPipeline, EulerAncestralDiscreteScheduler, StableDiffusionXLInpaintPipeline, ControlNetModel, UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from transformers import PretrainedConfig, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPTextModelWithProjection, CLIPConfig, CLIPTextConfig
from safetensors.torch import load_file as load_safetensors
from transUnet import get_new_key
from transVae import transform_line
from ccxx import convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint
from text_encoder_2_conver import convert_keys
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

    def __init__(self, model_path="stable-diffusion-xl-1.0-inpainting-0.1", controlnet_list_t = ['key_points'], use_out_test = False):
        if not hasattr(self, "_initialized"):
            print(f'start init sdxl inpainting')
            self.cache = {}  # 内存缓存字典
            self._initialized = True
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_path = model_path
            self.controlnet_list = controlnet_list_t #,'depth','canny','key_points','depth',,'seg'
            self.use_out_test = use_out_test
            self.text_device = self.device
            if self.use_out_test:
                self.text_device = "cpu"
            # 手动加载第一个文本编码器
            weights_path = "/mnt/sessd/civitai-downloader/juggernautXL_v9rdphoto2Inpaint.safetensors"
            if not weights_path or weights_path == '':
                print(f"no check point config")
                return
            # 加载预训练权重
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"权重文件未找到: {weights_path}")
            # 先加载配置文件
            state_dict = load_safetensors(weights_path)

            config_path = os.path.join(self.model_path, "text_encoder/config.json")
            config = CLIPConfig.from_json_file(config_path)
            text_encoder = CLIPTextModel(config).to(self.text_device)
            self.text_encoder = text_encoder
            print(f'start load test')
            self.load_pretrained_text_weights(state_dict)
            # 手动加载第二个文本编码器

            # 先加载配置文件
            config_path_2 = os.path.join(self.model_path, "text_encoder_2/config.json")
            config_2 = CLIPTextConfig.from_json_file(config_path_2)
            # 使用配置初始化模型
            text_encoder_2 = CLIPTextModelWithProjection(config=config_2).to(self.text_device)
            self.text_encoder_2 = text_encoder_2

            print(f'start load test_2')
            self.load_pretrained_text_2_weights(state_dict)

            # 手动加载第一个分词器
            tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer"))
            # 手动加载第二个分词器
            tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer_2"))

            # 使用配置初始化 VAE 模型
            # 手动初始化 AutoencoderKL，使用提供的配置参数
            self.vae = AutoencoderKL(
                act_fn="silu",
                block_out_channels=[128, 256, 512, 512],
                down_block_types=[
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D"
                ],
                in_channels=3,
                latent_channels=4,
                layers_per_block=2,
                norm_num_groups=32,
                out_channels=3,
                sample_size=512,
                scaling_factor=0.13025,
                up_block_types=[
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D"
                ]
            ).to(self.device)
            print(f'start load vae')
            self.load_pretrained_vae_weights(state_dict)
            print(f'start init unet')
            print("start init UNet2DConditionModel...")
            # 手动加载 UNet
            self.unet = UNet2DConditionModel.from_pretrained(os.path.join(self.model_path, "unet"), torch_dtype=torch.float16).to(self.device)
            print("UNet2DConditionModel initialized successfully.")

            print(f'start load unet')
            self.load_pretrained_unet_weights(state_dict)

            self.tokenizer = tokenizer
            self.tokenizer_2 = tokenizer_2
            self._print_component_load_info("Text Encoder 1", text_encoder)
            self._print_component_load_info("Text Encoder 2", text_encoder_2)

            print(f'start init pipe')
            del state_dict
            torch.cuda.empty_cache()
            gc.collect()
            self.pipe = self._initialize_pipe()

            # self.load_pretrained_weights(weights_path)

            # self.load_lora_weights(lora_path="/mnt/sessd/civitai-downloader/lora/biggunsxl_v11.safetensors", lora_scale=0.5, adapter_name="bigsxiiiixxl")
            # self.load_lora_weights(lora_path="/mnt/sessd/civitai-downloader/lora/SDXL_DatAss_v1.safetensors", lora_scale=0.7, adapter_name="datass")
            # self.load_lora_weights(lora_path="/mnt/sessd/civitai-downloader/lora/Realistic_Pussy_Xl-000010.safetensors", lora_scale=0.8, adapter_name="pppussy")


        print('suc lora')

    def _initialize_pipe(self):
        # 手动加载调度器
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), use_karras_sigmas=True)
        # scheduler = EulerDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), solver_order=3)

        # juggernautXL_v9rdphoto2Inpaint.safetensors 使用
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), use_karras_sigmas=True)
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"),  use_karras_sigmas=True)
        # scheduler = DDIMScheduler.from_pretrained(os.path.join(self.model_path, "text_encoder"))

        # lustifySDXLNSFW_v10-inpainting.safetensors 使用
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
        # scheduler.use_karras_sigmas = True

        if self.use_out_test:
            to_text_encoder = None
            to_text_encoder_2 = None
            to_tokenizer = None
            to_tokenizer_2 = None
        else:
            to_text_encoder = self.text_encoder
            to_text_encoder_2 = self.text_encoder_2
            to_tokenizer = self.tokenizer
            to_tokenizer_2 = self.tokenizer_2
        if len(self.controlnet_list) == 0:
            print(f'init ----------------------------- use Normal Pipeline '
                  f'{type(to_text_encoder)} {type(to_text_encoder_2)} {type(to_tokenizer)} {type(to_tokenizer_2)}')
            pipe = StableDiffusionXLInpaintPipeline(
                vae=self.vae,
                text_encoder=to_text_encoder,
                text_encoder_2=to_text_encoder_2,
                tokenizer=to_tokenizer,
                tokenizer_2=to_tokenizer_2,
                unet=self.unet,
                scheduler=scheduler
            )
            # pipe.to(self.device)
            return pipe
        controlnet_init_list = []
        for controlnet_item in self.controlnet_list:
            if 'key_points' == controlnet_item:
                # 加载 OpenPose 的 ControlNet 模型
                # controlnet_openpose = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16).to(self.device)
                # controlnet_init_list.append(controlnet_openpose)
                controlnet_openpose = ControlNetModel.from_pretrained("xinsir/controlnet-openpose-sdxl-1.0",torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet_openpose)
            elif 'depth' == controlnet_item:
                controlnet_dept = ControlNetModel.from_pretrained( "xinsir/controlnet-depth-sdxl-1.0",torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet_dept)
            elif 'seg' == controlnet_item:
                controlnet_seg = ControlNetModel.from_pretrained( "SargeZT/sdxl-controlnet-seg",torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet_seg)
            elif 'canny' == controlnet_item:
                controlnet_canny = ControlNetModel.from_pretrained( "xinsir/controlnet-canny-sdxl-1.0",torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet_canny)
        # 创建自定义管道
        if len(controlnet_init_list) == 0:
            to_controlnet = None
        elif len(controlnet_init_list) == 1:
            to_controlnet = controlnet_init_list[0]
        else:
            to_controlnet = controlnet_init_list
        print(f'init ----------------------------- use ControlNet Pipeline '
              f'{to_controlnet} {type(to_text_encoder)} {type(to_text_encoder_2)} {type(to_tokenizer)} {type(to_tokenizer_2)}')
        pipe = StableDiffusionXLControlNetInpaintPipeline(
            controlnet=to_controlnet, #[controlnet_openpose, controlnet_normal],
            vae=self.vae,
            text_encoder=to_text_encoder,
            text_encoder_2=to_text_encoder_2,
            tokenizer=to_tokenizer,
            tokenizer_2=to_tokenizer_2,
            unet=self.unet,
            scheduler=scheduler
        )
        # pipe.to(self.device)
        return pipe

    def load_pretrained_unet_weights(self, state_dict):
        # 替换 key
        unet_state_dict = convert_ldm_unet_checkpoint(state_dict)
        unet_model_dict = self.unet.state_dict()
        loaded_unet = {}
        # 仅加载形状匹配的键
        for key in unet_state_dict.keys():
            if key in unet_model_dict:
                if unet_state_dict[key].shape == unet_model_dict[key].shape:
                    loaded_unet[key] = unet_state_dict[key]
                else:
                    print(f"this {key} len shape {unet_state_dict[key].shape} and sa shape {unet_model_dict[key].shape}")

        print(f"this len is {len(state_dict.keys())}")
        # 打印 state_dict 的所有键
        self._write_keys_to_file("weights_keys.txt", state_dict.keys())
        # 加载 UNet 权重
        print("Loading UNet weights...")
        unet_keys = unet_model_dict.keys()

        unet_load_info = self.unet.load_state_dict(loaded_unet, strict=False)
        print(f"UNet total keys: {len(unet_keys)}, loaded keys: {len(loaded_unet)}, missing keys: {len(unet_load_info.missing_keys)}")
        print(f"Missing UNet keys: {unet_load_info.missing_keys}\n")
        self._write_keys_to_file("Missingunet.txt", unet_load_info.missing_keys)


    def load_pretrained_vae_weights(self, state_dict):
        vae_state_dict = replace_keys_vae(state_dict)
        # 存储所有 num_head_channels 的值
        num_head_channels_list = []

        # 遍历所有的注意力权重
        for key, value in vae_state_dict.items():
            if "attentions" in key and ("to_q.weight" in key or "to_k.weight" in key or "to_v.weight" in key):
                # 获取第一个维度的大小
                num_heads = value.shape[0]
                # 获取 attention 的 channels 大小
                num_head_channels = num_heads // 3  # 通常 attention head 数是 channels 的三倍
                num_head_channels_list.append(num_head_channels)

        # 如果有多种 num_head_channels，可以打印出所有值
        print("All calculated num_head_channels:", num_head_channels_list)

        # 取一个示例（或统一）作为 config["num_head_channels"]
        if num_head_channels_list:
            config_num_head_channels = num_head_channels_list[0]  # 示例值
            print("config['num_head_channels']:", config_num_head_channels)

        # 加载 VAE 权重
        print("Loading VAE weights...")
        vae_model_dict = self.vae.state_dict()  # 获取完整的 VAE 模型状态字典
        loaded_vae = {}

        # 遍历所有的注意力权重
        for key, value in vae_model_dict.items():
            if "attentions" in key and ("to_q.weight" in key or "to_k.weight" in key or "to_v.weight" in key):
                # 获取第一个维度的大小
                num_heads = value.shape[0]
                # 获取 attention 的 channels 大小
                num_head_channels = num_heads // 3  # 通常 attention head 数是 channels 的三倍
                num_head_channels_list.append(num_head_channels)

        # 如果有多种 num_head_channels，可以打印出所有值
        print("All calculated num_head_channels:", num_head_channels_list)

        # 取一个示例（或统一）作为 config["num_head_channels"]
        if num_head_channels_list:
            config_num_head_channels = num_head_channels_list[0]  # 示例值
            print("config['num_head_channels']:", config_num_head_channels)

        vae_state_dict = convert_ldm_vae_checkpoint(state_dict)
        # 仅加载形状匹配的键
        for key in vae_state_dict.keys():
            if key in vae_model_dict:
                if vae_state_dict[key].shape == vae_model_dict[key].shape:
                    loaded_vae[key] = vae_state_dict[key]
                else:
                    print(f"this {key} vae shape {vae_state_dict[key].shape} and vae shape {vae_model_dict[key].shape}")
        vae_load_info = self.vae.load_state_dict(loaded_vae, strict=False)
        print(f"VAE total keys: {len(vae_model_dict.keys())}, loaded keys: {len(loaded_vae)}, missing keys: {len(vae_load_info.missing_keys)}")
        print(f"Missing VAE keys: {vae_load_info.missing_keys}\n")

        self._write_keys_to_file("MissinguVAE.txt", vae_load_info.missing_keys)


    def load_pretrained_text_weights(self, state_dict):
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
        text_encoder_keys = self.text_encoder.state_dict().keys()
        loaded_text_encoder = modified_state_dict
        text_encoder_load_info = self.text_encoder.load_state_dict(loaded_text_encoder, strict=False)

        print(f"Text encoder total keys: {len(text_encoder_keys)}, loaded keys: {len(loaded_text_encoder)}, missing keys: {len(text_encoder_load_info.missing_keys)}")
        print(f"Missing text encoder keys: {text_encoder_load_info.missing_keys}\n")

        # 加载文本编码器权重
        print("Loading text encoder 2 weights...")
        # 移除键名中的 'conditioner.embedders.0.transformer.' 前缀
        print("Loading text encoder 2 weights...")


    def load_pretrained_text_2_weights(self, state_dict):

        text_encoder_keys = self.text_encoder_2.state_dict().keys()

        loaded_text_encoder = {}

        text_2_model_dict = self.text_encoder_2.state_dict()
        # 仅加载形状匹配的键
        modified_state_dict = convert_keys(state_dict)
        for key in modified_state_dict.keys():
            if key in text_2_model_dict:
                if modified_state_dict[key].shape == text_2_model_dict[key].shape:
                    loaded_text_encoder[key] = modified_state_dict[key]
                else:
                    print(f"keys shape: {key} checkpoint shape is {modified_state_dict[key].shape} but model shape is {text_2_model_dict[key].shape}\n")

        text_encoder_load_info = self.text_encoder_2.load_state_dict(loaded_text_encoder, strict=False)
        print(f"Text encoder 2 total keys: {len(text_encoder_keys)}, loaded keys: {len(loaded_text_encoder)}, missing keys: {len(text_encoder_load_info.missing_keys)}")
        print(f"Missing text encoder 2 keys: {text_encoder_load_info.missing_keys}\n")
        self._write_keys_to_file("Missinguencoder2.txt", text_encoder_load_info.missing_keys)


    def __upload_pip__(self):
        if self._instance is not None:
            if self.pipe:
                del self.pipe  # 删除模型对象
            CustomInpaintPipeline._instance = None  # 重置单例实例

    def release_resources(self):
        # 清理占用的资源
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()  # 释放GPU内存
        print('Resources released')

        # 将单例重置为None
        CustomInpaintPipeline._instance = None

    def __del__(self):
        # 实例销毁时自动释放资源
        self.release_resources()

    def _write_keys_to_file(self, filename, keys):
        # with open(filename, 'w') as f:
        #     for key in keys:
        #         f.write(f"{key}\n")
        return

    def _generate_unique_key(self, text):
        # 为每个文本生成唯一键
        unique_key = hashlib.sha256(text.encode()).hexdigest()
        return unique_key

    def encode_and_cache(self, prompt, prompt_2, reverse_prompt, reverse_prompt_2):
        if self.use_out_test:
            return encode_prompt_with_cache(self.pipe, prompt, prompt_2, reverse_prompt, reverse_prompt_2, self.tokenizer, self.tokenizer_2, self.text_encoder, self.text_encoder_2)
        return (None, None, None, None)
    def _print_component_load_info(self, component_name, component):
        state_dict = component.state_dict()
        total_keys = len(state_dict.keys())
        loaded_keys = sum(1 for k, v in state_dict.items() if v is not None)
        missing_keys = total_keys - loaded_keys

        print(f"{component_name} total keys: {total_keys}, loaded keys: {loaded_keys}, missing keys: {missing_keys}")

    def load_pretrained_weights(self, weights_path, load_part_list = ['unet','vae','t1','t2']):
        if not weights_path or weights_path == '':
            print(f"no check point config")
            return
        # 加载预训练权重
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"权重文件未找到: {weights_path}")

        state_dict = load_safetensors(weights_path)

        # 替换 key
        unet_state_dict = convert_ldm_unet_checkpoint(state_dict)
        unet_model_dict = self.pipe.unet.state_dict()
        loaded_unet = {}
        # 仅加载形状匹配的键
        for key in unet_state_dict.keys():
            if key in unet_model_dict:
                if unet_state_dict[key].shape == unet_model_dict[key].shape:
                    loaded_unet[key] = unet_state_dict[key]
                else:
                    print(f"this {key} len shape {unet_state_dict[key].shape} and sa shape {unet_model_dict[key].shape}")

        print(f"this {weights_path} len is {len(state_dict.keys())}")
        # 打印 state_dict 的所有键
        self._write_keys_to_file("weights_keys.txt", state_dict.keys())
        # 加载 UNet 权重
        print("Loading UNet weights...")
        unet_keys = unet_model_dict.keys()
        if 'unet' in load_part_list:
            unet_load_info = self.pipe.unet.load_state_dict(loaded_unet, strict=False)
            print(f"UNet total keys: {len(unet_keys)}, loaded keys: {len(loaded_unet)}, missing keys: {len(unet_load_info.missing_keys)}")
            print(f"Missing UNet keys: {unet_load_info.missing_keys}\n")
            self._write_keys_to_file("Missingunet.txt", unet_load_info.missing_keys)

        vae_state_dict = replace_keys_vae(state_dict)

        # 存储所有 num_head_channels 的值
        num_head_channels_list = []

        # 遍历所有的注意力权重
        for key, value in vae_state_dict.items():
            if "attentions" in key and ("to_q.weight" in key or "to_k.weight" in key or "to_v.weight" in key):
                # 获取第一个维度的大小
                num_heads = value.shape[0]
                # 获取 attention 的 channels 大小
                num_head_channels = num_heads // 3  # 通常 attention head 数是 channels 的三倍
                num_head_channels_list.append(num_head_channels)

        # 如果有多种 num_head_channels，可以打印出所有值
        print("All calculated num_head_channels:", num_head_channels_list)

        # 取一个示例（或统一）作为 config["num_head_channels"]
        if num_head_channels_list:
            config_num_head_channels = num_head_channels_list[0]  # 示例值
            print("config['num_head_channels']:", config_num_head_channels)

        # 加载 VAE 权重
        print("Loading VAE weights...")
        vae_model_dict = self.pipe.vae.state_dict()  # 获取完整的 VAE 模型状态字典
        loaded_vae = {}

        # 遍历所有的注意力权重
        for key, value in vae_model_dict.items():
            if "attentions" in key and ("to_q.weight" in key or "to_k.weight" in key or "to_v.weight" in key):
                # 获取第一个维度的大小
                num_heads = value.shape[0]
                # 获取 attention 的 channels 大小
                num_head_channels = num_heads // 3  # 通常 attention head 数是 channels 的三倍
                num_head_channels_list.append(num_head_channels)

        # 如果有多种 num_head_channels，可以打印出所有值
        print("All calculated num_head_channels:", num_head_channels_list)

        # 取一个示例（或统一）作为 config["num_head_channels"]
        if num_head_channels_list:
            config_num_head_channels = num_head_channels_list[0]  # 示例值
            print("config['num_head_channels']:", config_num_head_channels)

        vae_state_dict = convert_ldm_vae_checkpoint(state_dict)
        # 仅加载形状匹配的键
        for key in vae_state_dict.keys():
            if key in vae_model_dict:
                if vae_state_dict[key].shape == vae_model_dict[key].shape:
                    loaded_vae[key] = vae_state_dict[key]
                else:
                    print(f"this {key} vae shape {vae_state_dict[key].shape} and vae shape {vae_model_dict[key].shape}")
        if 'vae' in load_part_list:
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
        text_encoder_keys = self.text_encoder.state_dict().keys()
        loaded_text_encoder = modified_state_dict
        if 't1' in load_part_list:
            text_encoder_load_info = self.text_encoder.load_state_dict(loaded_text_encoder, strict=False)

            print(f"Text encoder total keys: {len(text_encoder_keys)}, loaded keys: {len(loaded_text_encoder)}, missing keys: {len(text_encoder_load_info.missing_keys)}")
            print(f"Missing text encoder keys: {text_encoder_load_info.missing_keys}\n")

            # 加载文本编码器权重
            print("Loading text encoder 2 weights...")
            # 移除键名中的 'conditioner.embedders.0.transformer.' 前缀
            print("Loading text encoder 2 weights...")

        text_encoder_keys = self.text_encoder_2.state_dict().keys()

        loaded_text_encoder = {}

        text_2_model_dict = self.text_encoder_2.state_dict()
        # 仅加载形状匹配的键
        modified_state_dict = convert_keys(state_dict)
        for key in modified_state_dict.keys():
            if key in text_2_model_dict:
                if modified_state_dict[key].shape == text_2_model_dict[key].shape:
                    loaded_text_encoder[key] = modified_state_dict[key]
                else:
                    print(f"keys shape: {key} checkpoint shape is {modified_state_dict[key].shape} but model shape is {text_2_model_dict[key].shape}\n")
        if 't2' in load_part_list:
            text_encoder_load_info = self.text_encoder_2.load_state_dict(loaded_text_encoder, strict=False)
            print(
                f"Text encoder 2 total keys: {len(text_encoder_keys)}, loaded keys: {len(loaded_text_encoder)}, missing keys: {len(text_encoder_load_info.missing_keys)}")
            print(f"Missing text encoder 2 keys: {text_encoder_load_info.missing_keys}\n")

            self._write_keys_to_file("Missinguencoder2.txt", text_encoder_load_info.missing_keys)
    def unload_lora_weights(self):
        print("Un Loading LoRA layers:")
        self.pipe.unload_lora_weights()
        print(f"Un loaded lora")

    def load_lora_weights(self, lora_path=None, adapter_name=None, lora_scale=0.7):
        print("Loaded LoRA layers:")
        pretrained_model_name_or_path_or_dict = lora_path
        if isinstance(lora_path, dict):
            pretrained_model_name_or_path_or_dict = lora_path.copy()
        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas = self.pipe.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.pipe.unet.config, force_download=True
        )
        is_correct_format = all("lora" in key or "dora_scale" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.pipe.load_lora_into_unet(
            state_dict, network_alphas=network_alphas, unet=self.pipe.unet, adapter_name=adapter_name, _pipeline=self.pipe
        )
        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.pipe.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.pipe.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self.pipe,
            )
        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.pipe.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=self.pipe.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self.pipe,
            )
        print(f"Loaded {len(state_dict)} layers of LoRA weights into UNet.")
        print(f"Loaded {len(text_encoder_state_dict)} layers of LoRA weights into text_encoder.")
        print(f"Loaded {len(text_encoder_2_state_dict)} layers of LoRA weights into text_encoder_2.")

        print(f"loaded lora path: {lora_path}\n")

    def load_lora_weights_array(self, lora_configs=None):
        for lora_config in lora_configs:
            self.load_lora_weights(lora_config['path'], lora_config['name'])
        adapter_list = []
        adapter_weight_list = []
        for lora_config in lora_configs:
            adapter_list.append(lora_config['name'])
            adapter_weight_list.append(lora_config['weight'])
        self.pipe.set_adapters(adapter_list, adapter_weights=adapter_weight_list)

    def generate_image(self, prompt="nude,high quality,real skin details",
                       prompt_2="high-waisted, natural asses, big and natural boobs",
                       reverse_prompt="deformed, bad anatomy, mutated, long neck, narrow Hips",
                       reverse_prompt_2="disconnected limbs,unnaturally contorted position, unnaturally thin waist",
                       init_image=None, mask_image=None, num_inference_steps = 30, seed = 178, guidance_scale = 6,
                       strength=0.9, control_image=None, controlnet_conditioning_scale=1.0):
        torch.manual_seed(seed)
        result = self.pipe(
                prompt = prompt,
                prompt_2 = prompt_2,
                negative_prompt= reverse_prompt,
                negative_prompt_2 = reverse_prompt_2,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                # callback=progress_callback,
                strength=strength,
                control_image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                # padding_mask_crop=32,
                callback_steps=1  # 调用回调的步数
            ).images[0]
        # 释放 GPU 内存
        if 'init_image' in locals():
            del init_image
        else:
            print("init_image does not exist.")
        if 'mask_image' in locals():
            del mask_image
        else:
            print("mask_image does not exist.")
        torch.cuda.empty_cache()
        gc.collect()
        return result



