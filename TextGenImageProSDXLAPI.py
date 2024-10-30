from PIL import Image
import os
import traceback
import hashlib, requests, random
import torch
import uuid
import gc
import zipfile
from CustCoun import encode_prompt_with_cache
from CoreService import GenContextParam
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
from diffusers import (StableDiffusionXLPipeline,StableDiffusionXLControlNetPipeline, EulerAncestralDiscreteScheduler,
                       DEISMultistepScheduler, LMSDiscreteScheduler, HeunDiscreteScheduler, RePaintScheduler,
                       ControlNetModel, UNet2DConditionModel, AutoencoderKL,
                       EulerDiscreteScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, DDIMScheduler)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPTextModelWithProjection
from safetensors.torch import load_file as load_safetensors
from TextGenImageProFluxAPI import gen_img_canny_control_with_face, gen_img_and_swap_face
from transUnet import get_new_key
from transVae import transform_line
from ccxx import convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint
from text_encoder_2_conver import convert_keys
from book_yes_logger_config import logger
from CoreService import req_replace_face
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

    def __init__(self, model_path="/nvme0n1-disk/ipchange/stable-diffusion-xl-base-1.0", controlnet_list_t = [], use_out_test = False, use_noise = False):
        if not hasattr(self, "_initialized"):
            self.cache = {}  # 内存缓存字典
            self._initialized = True
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_path = model_path
            self.controlnet_list = controlnet_list_t #,'depth','canny','key_points','depth',,'seg'
            self.use_out_test = use_out_test
            self.use_noise = use_noise
            self.text_device = self.device
            if self.use_out_test:
                self.text_device = "cpu"
            # 手动加载第一个文本编码器
            text_encoder = CLIPTextModel.from_pretrained(os.path.join(self.model_path, "text_encoder"), torch_dtype=torch.float16).to(self.text_device)
            # 手动加载第二个文本编码器
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(os.path.join(self.model_path, "text_encoder_2"), torch_dtype=torch.float16).to(self.text_device)
            # 手动加载第一个分词器
            tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer"))
            # 手动加载第二个分词器
            tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer_2"))
            # text_encoder = text_encoder.half()
            # text_encoder_2 = text_encoder_2.half()

            self.text_encoder = text_encoder
            self.text_encoder_2 = text_encoder_2
            self.tokenizer = tokenizer
            self.tokenizer_2 = tokenizer_2
            self._print_component_load_info("Text Encoder 1", text_encoder)
            self._print_component_load_info("Text Encoder 2", text_encoder_2)
            self.pipe = self._initialize_pipe()
            weights_path = "/nvme0n1-disk/civitai-downloader/STOIQOAfroditeFLUXXL_XL31.safetensors"
            self.load_pretrained_weights(weights_path)
            # self.load_lora_weights()
        logger.info('suc lora')

    def release_resources(self):
        # 清理占用的资源
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()  # 释放GPU内存
        logger.info('Resources released')

        # 将单例重置为None
        CustomInpaintPipeline._instance = None

    def __del__(self):
        # 实例销毁时自动释放资源
        self.release_resources()

    def _initialize_pipe(self):
        # 手动加载调度器
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), use_karras_sigmas=True)
        # 经典
        # scheduler = EulerDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        # 随机性高
        #EulerAncestralDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        # scheduler = UniPCMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        # 不稳定
        # scheduler = LMSDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        # juggernautXL_v9rdphoto2Inpaint.safetensors 使用 LMSDiscreteScheduler, or PNDMScheduler.
        # 高质量
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), solver_order=2, algorithm_type = "sde-dpmsolver++", use_karras_sigmas=True)
        logger.info(f"being scheduler init")
        if self.use_noise:
            scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )
        else:
            scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), solver_order=2, algorithm_type = "sde-dpmsolver++", use_karras_sigmas=True)


        # contrl
        # 自然
        # scheduler = EulerAncestralDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        # 满 ，还一些
        # scheduler = HeunDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        # scheduler = DEISMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        # scheduler = RePaintScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), use_karras_sigmas=True)
        # 初始化 RePaintScheduler
        # scheduler = RePaintScheduler(
        #     num_train_timesteps=1000,  # 默认扩散步骤数
        #     beta_start=0.0001,  # 开始的 beta 值
        #     beta_end=0.02,  # 结束的 beta 值
        #     beta_schedule="linear",  # 使用线性调度
        #     eta=0.0  # DDIM 模式下的噪声权重
        # )
        # scheduler = DDIMScheduler.from_pretrained(os.path.join(self.model_path, "text_encoder"))
        # lustifySDXLNSFW_v10-inpainting.safetensors 使用
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), solver_order = 3, algorithm_type="dpmsolver++", use_karras_sigmas=True)
        # scheduler.use_karras_sigmas = True
        # # 手动加载 VAE
        # noise_scheduler = DDIMScheduler(
        #     num_train_timesteps=1000,
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     clip_sample=False,
        #     set_alpha_to_one=False,
        #     steps_offset=1,
        # )
        # vae_model_path = "stabilityai/sd-vae-ft-mse"
        # vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=torch.float16).to(self.device)
        logger.info(f"being vae init")
        vae = AutoencoderKL.from_pretrained(os.path.join(self.model_path, "vae"), torch_dtype=torch.float16).to(self.device)
        # 手动加载 UNet
        unet = UNet2DConditionModel.from_pretrained(os.path.join(self.model_path, "unet"), torch_dtype=torch.float16).to(self.device)
        # # 将各组件的键写入文件
        self._write_keys_to_file("vae_keys.txt", vae.state_dict().keys())
        self._write_keys_to_file("unet_keys.txt", unet.state_dict().keys())

        # 设置半精度
        # vae = vae.half()
        # unet = unet.half()
        # 打印各组件的加载信息
        self._print_component_load_info("UNet", unet)
        self._print_component_load_info("VAE", vae)

        logger.info(f"total keys: {len(unet.state_dict().keys()) + len(vae.state_dict().keys())} "
              f"text_encoder keys:  {len(self.text_encoder.state_dict().keys())} "
              f"text_encoder_2 keys: {len(self.text_encoder_2.state_dict().keys())}"
              )

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
            logger.info(f'init ----------------------------- use Normal Pipeline '
                  f'{type(to_text_encoder)} {type(to_text_encoder_2)} {type(to_tokenizer)} {type(to_tokenizer_2)}')
            pipe = StableDiffusionXLPipeline(
                vae=vae,
                text_encoder=to_text_encoder,
                text_encoder_2=to_text_encoder_2,
                tokenizer=to_tokenizer,
                tokenizer_2=to_tokenizer_2,
                unet=unet,
                scheduler=scheduler,
                add_watermarker=False
            )
            # pipe.register_to_config(requires_aesthetics_score=True)
            pipe.to(self.device)
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
                # controlnet_dept = ControlNetModel.from_pretrained("diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16).to(self.device)
                # controlnet_init_list.append(controlnet_dept)
                controlnet_dept = ControlNetModel.from_pretrained( "xinsir/controlnet-depth-sdxl-1.0",torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet_dept)
            elif 'seg' == controlnet_item:
                controlnet_seg = ControlNetModel.from_pretrained( "SargeZT/sdxl-controlnet-seg",torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet_seg)
            elif 'canny' == controlnet_item:
                controlnet_canny = ControlNetModel.from_pretrained( "xinsir/controlnet-scribble-sdxl-1.0",torch_dtype=torch.float16).to(self.device)
                controlnet_init_list.append(controlnet_canny)
        # 创建自定义管道
        if len(controlnet_init_list) == 0:
            to_controlnet = None
        elif len(controlnet_init_list) == 1:
            to_controlnet = controlnet_init_list[0]
        else:
            to_controlnet = controlnet_init_list
        logger.info(f'init ----------------------------- use ControlNet Pipeline '
              f'{len(controlnet_init_list)} {type(to_text_encoder)} {type(to_text_encoder_2)} {type(to_tokenizer)} {type(to_tokenizer_2)}')
        pipe = StableDiffusionXLControlNetPipeline(
            controlnet=to_controlnet, #[controlnet_openpose, controlnet_normal],
            vae=vae,
            text_encoder=to_text_encoder,
            text_encoder_2=to_text_encoder_2,
            tokenizer=to_tokenizer,
            tokenizer_2=to_tokenizer_2,
            unet=unet,
            scheduler=scheduler
        )
        # pipe.register_to_config(requires_aesthetics_score=True)
        pipe.to(self.device)
        return pipe

    def __upload_pip__(self):
        if self._instance is not None:
            if self.pipe:
                del self.pipe  # 删除模型对象
            CustomInpaintPipeline._instance = None  # 重置单例实例

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

        logger.info(f"{component_name} total keys: {total_keys}, loaded keys: {loaded_keys}, missing keys: {missing_keys}")

    def load_pretrained_weights(self, weights_path, load_part_list = ['unet','vae','t1','t2']):
        if not weights_path or weights_path == '':
            logger.info(f"no check point config")
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
                    logger.info(f"this {key} len shape {unet_state_dict[key].shape} and sa shape {unet_model_dict[key].shape}")

        logger.info(f"this {weights_path} len is {len(state_dict.keys())}")
        # 打印 state_dict 的所有键
        self._write_keys_to_file("weights_keys.txt", state_dict.keys())
        # 加载 UNet 权重
        logger.info("Loading UNet weights...")
        unet_keys = unet_model_dict.keys()
        if 'unet' in load_part_list:
            unet_load_info = self.pipe.unet.load_state_dict(loaded_unet, strict=False)
            logger.info(f"UNet total keys: {len(unet_keys)}, loaded keys: {len(loaded_unet)}, missing keys: {len(unet_load_info.missing_keys)}")
            logger.info(f"Missing UNet keys: {unet_load_info.missing_keys}\n")
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
        logger.info(f"All calculated num_head_channels:{num_head_channels_list}")

        # 取一个示例（或统一）作为 config["num_head_channels"]
        if num_head_channels_list:
            config_num_head_channels = num_head_channels_list[0]  # 示例值
            logger.info(f"config['num_head_channels']:{config_num_head_channels}")

        # 加载 VAE 权重
        logger.info("Loading VAE weights...")
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
        logger.info(f"All calculated num_head_channels:{num_head_channels_list}")

        # 取一个示例（或统一）作为 config["num_head_channels"]
        if num_head_channels_list:
            config_num_head_channels = num_head_channels_list[0]  # 示例值
            logger.info(f"config['num_head_channels']:{config_num_head_channels}")

        vae_state_dict = convert_ldm_vae_checkpoint(state_dict)
        # 仅加载形状匹配的键
        for key in vae_state_dict.keys():
            if key in vae_model_dict:
                if vae_state_dict[key].shape == vae_model_dict[key].shape:
                    loaded_vae[key] = vae_state_dict[key]
                else:
                    logger.info(f"this {key} vae shape {vae_state_dict[key].shape} and vae shape {vae_model_dict[key].shape}")
        if 'vae' in load_part_list:
            vae_load_info = self.pipe.vae.load_state_dict(loaded_vae, strict=False)
            logger.info(
            f"VAE total keys: {len(vae_model_dict.keys())}, loaded keys: {len(loaded_vae)}, missing keys: {len(vae_load_info.missing_keys)}")
            logger.info(f"Missing VAE keys: {vae_load_info.missing_keys}\n")

            self._write_keys_to_file("MissinguVAE.txt", vae_load_info.missing_keys)

        # 加载文本编码器权重
        logger.info("Loading text encoder weights...")
        # 移除键名中的 'conditioner.embedders.0.transformer.' 前缀
        modified_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('conditioner.embedders.0.transformer.'):
                # 删除前缀，保留后续部分
                new_key = k.replace('conditioner.embedders.0.transformer.', '')
                modified_state_dict[new_key] = v
            else:
                modified_state_dict[k] = v
        logger.info("Loading text encoder weights...")
        text_encoder_keys = self.text_encoder.state_dict().keys()
        loaded_text_encoder = modified_state_dict
        if 't1' in load_part_list:
            text_encoder_load_info = self.text_encoder.load_state_dict(loaded_text_encoder, strict=False)

            logger.info(f"Text encoder total keys: {len(text_encoder_keys)}, loaded keys: {len(loaded_text_encoder)}, missing keys: {len(text_encoder_load_info.missing_keys)}")
            logger.info(f"Missing text encoder keys: {text_encoder_load_info.missing_keys}\n")

            # 加载文本编码器权重
            logger.info("Loading text encoder 2 weights...")
            # 移除键名中的 'conditioner.embedders.0.transformer.' 前缀
            logger.info("Loading text encoder 2 weights...")

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
                    logger.info(f"keys shape: {key} checkpoint shape is {modified_state_dict[key].shape} but model shape is {text_2_model_dict[key].shape}\n")
        if 't2' in load_part_list:
            text_encoder_load_info = self.text_encoder_2.load_state_dict(loaded_text_encoder, strict=False)
            logger.info(
                f"Text encoder 2 total keys: {len(text_encoder_keys)}, loaded keys: {len(loaded_text_encoder)}, missing keys: {len(text_encoder_load_info.missing_keys)}")
            logger.info(f"Missing text encoder 2 keys: {text_encoder_load_info.missing_keys}\n")

            self._write_keys_to_file("Missinguencoder2.txt", text_encoder_load_info.missing_keys)
    def unload_lora_weights(self):
        logger.info("Un Loading LoRA layers:")
        self.pipe.unload_lora_weights()
        logger.info(f"Un loaded lora")

    def load_lora_weights(self, lora_path=None, adapter_name=None, lora_scale=0.7):
        logger.info("Loaded LoRA layers:")
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
        logger.info(f"Loaded {len(state_dict)} layers of LoRA weights into UNet.")
        logger.info(f"Loaded {len(text_encoder_state_dict)} layers of LoRA weights into text_encoder.")
        logger.info(f"Loaded {len(text_encoder_2_state_dict)} layers of LoRA weights into text_encoder_2.")

        logger.info(f"loaded lora path: {lora_path}\n")

    def load_lora_weights_array(self, lora_configs=None):
        for lora_config in lora_configs:
            self.load_lora_weights(lora_config['path'], lora_config['name'])
        adapter_list = []
        adapter_weight_list = []
        for lora_config in lora_configs:
            adapter_list.append(lora_config['name'])
            adapter_weight_list.append(lora_config['weight'])
        self.pipe.set_adapters(adapter_list, adapter_weights=adapter_weight_list)

import cv2
from insightface.app import FaceAnalysis
import torch
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from PIL import Image
from insightface.utils import face_align
from ip_adapter import IPAdapterPlusXL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL, IPAdapterFaceIDPlusXL, IPAdapterFaceID

def req_text_gen(progress, had, room_id, port=5003):
    # 发起 HTTP POST 请求
    url = f"http://localhost:{port}/"
    data = {
        'progress': progress,
        'had': had,
        'room_id':room_id
    }
    try:
        response = requests.post(url+'gencallback', data=data)
    except Exception as e:
        logger.error(f"Error processing response: {e}")

def call_main_service(progress, had, room_id):
    logger.info(f'call back {progress} {had} {room_id}')
    req_text_gen(progress, had, room_id)
    return
def progress_callback(step, t, latents, had, room_id, num_inference_steps):
    progress = int((step / num_inference_steps) * 100)
    call_main_service(progress, had, room_id)

# 定义回调函数并传递 'had' 参数
def create_callback(had_value, room_id, num_inference_steps):
    return lambda step, t, latents: progress_callback(step, t, latents, had_value, room_id, num_inference_steps)

def center_image_on_black_background(image, target_size=(640, 640)):
    """
    如果图像小于指定大小，则将其放在纯黑色背景的中心。

    参数:
    - image: 要处理的图像（作为 NumPy 数组）。
    - target_size: 背景图像的目标尺寸，默认值为 (640, 640)。

    返回:
    - 处理后的图像，大小为 target_size。
    """
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # 如果图像已经大于等于目标大小，直接返回原图像
    if original_width >= target_width and original_height >= target_height:
        return image

    # 创建一个纯黑色的背景图像
    black_background = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 计算图像应该放置的位置，使其居中
    x_offset = (target_width - original_width) // 2
    y_offset = (target_height - original_height) // 2

    # 将原图像粘贴到黑色背景上
    black_background[y_offset:y_offset + original_height, x_offset:x_offset + original_width] = image

    return black_background

def ip_sdxl(face_pic_path="", prompt="", negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, blurry"):

    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    image = cv2.imread(face_pic_path)
    faces = app.get(image)

    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    ip_ckpt = "/nvme0n1-disk/ipchange/IP-Adapter/IP-Adapter/hub_sdxl_models/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin"
    device = "cuda"

    sdxlPipe = CustomInpaintPipeline()
    pipe = sdxlPipe.pipe
    # load ip-adapter
    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=2,
        width=1024, height=1024,scale=0.8,
        num_inference_steps=30, guidance_scale=7.5, seed=2023
    )
    del ip_model
    # 释放 GPU 缓存
    torch.cuda.empty_cache()
    # 触发垃圾回收，释放未使用的内存
    gc.collect()
    return images
def ip_sdxl_plus_v1(face_pic_path="", prompt="", negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, blurry", seed=2023, only_face_pic_path="", room_id=None):
    if only_face_pic_path is not None and only_face_pic_path != '':
        face_pic_path = only_face_pic_path
        image = cv2.imread(face_pic_path)
        image = center_image_on_black_background(image)
    else:
        image = cv2.imread(face_pic_path)
    logger.info(f'user face img is {face_pic_path}')
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(image)
    face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)  # you can also segment the face
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_pil_image = Image.fromarray(face_image_rgb)

    # base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    image_encoder_path = "/nvme0n1-disk/ipchange/IP-Adapter/IP-Adapter/models/image_encoder"
    # image_encoder_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    ip_ckpt = "/nvme0n1-disk/ipchange/IP-Adapter/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin"
    device = "cuda"
    sdxlPipe = CustomInpaintPipeline()
    pipe = sdxlPipe.pipe
    logger.info(f'finish load sdxl .......')
    # load ip-adapter
    ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    logger.info(f'gen it prompt: {prompt} negative_prompt: {negative_prompt}')
    images = ip_model.generate(pil_image=face_pil_image, num_samples=4, num_inference_steps=30, seed=seed,
                               callback=create_callback('3/7', room_id, 30),callback_steps=1,
                               prompt=prompt, negative_prompt=negative_prompt)
    del ip_model
    # 释放 GPU 缓存
    torch.cuda.empty_cache()
    # 触发垃圾回收，释放未使用的内存
    gc.collect()
    return images

def ip_sdxl_plus_v2(face_pic_path="", prompt="", negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, blurry", seed=2023, only_face_pic_path="", room_id=None):
    if only_face_pic_path is not None and only_face_pic_path != '':
        face_pic_path = only_face_pic_path
        image = cv2.imread(face_pic_path)
        image = center_image_on_black_background(image)
    else:
        image = cv2.imread(face_pic_path)
    logger.info(f'user face img is {face_pic_path}')
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(image)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)  # you can also segment the face
    # base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # image_encoder_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    ip_ckpt = "/nvme0n1-disk/ipchange/IP-Adapter/IP-Adapter/hub_sdxl_models/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sdxl.bin"
    device = "cuda"
    sdxlPipe = CustomInpaintPipeline()
    pipe = sdxlPipe.pipe
    logger.info(f'finish load sdxl .......')
    # load ip-adapter
    try:
        ip_model = IPAdapterFaceIDPlusXL(pipe, image_encoder_path, ip_ckpt, device)
    except Exception as e:
        # 打印完整的错误堆栈信息
        print("Error occurred:")
        traceback.print_exc()
    logger.info(f'gen it prompt: {prompt} negative_prompt: {negative_prompt}')
    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, face_image=face_image, faceid_embeds=faceid_embeds,
        s_scale=1.0,scale=1.0, callback=create_callback('1/7', room_id, 60),callback_steps=2,
        num_samples=2 , width=1024, height=1024, num_inference_steps=60, seed=seed
    )
    del ip_model
    # 释放 GPU 缓存
    torch.cuda.empty_cache()
    # 触发垃圾回收，释放未使用的内存
    gc.collect()
    return images

def ip_sdxl_work_plus(face_pic_path="", prompt="", negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, blurry"):
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    image = cv2.imread(face_pic_path)
    faces = app.get(image)

    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)  # you can also segment the face

    v2 = True
    # base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    image_encoder_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    ip_ckpt = "/nvme0n1-disk/ipchange/IP-Adapter/IP-Adapter/hub_sdxl_models/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sdxl.bin"
    device = "cuda"
    base_model_path = "SG161222/RealVisXL_V3.0"
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )

    logger.info(f'finish load sdxl .......')
    # load ip-adapter
    ip_model = IPAdapterFaceIDPlusXL(pipe, image_encoder_path, ip_ckpt, device)

    logger.info(f'gen it prompt: {prompt} negative_prompt: {negative_prompt}')
    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, face_image=face_image, faceid_embeds=faceid_embeds, shortcut=v2,
        s_scale=1.0,
        num_samples=4 , width=1024, height=1024, num_inference_steps=30, seed=2023
    )
    # 删除模型和管道对象
    del ip_model
    del pipe
    del app
    # 释放 GPU 缓存
    torch.cuda.empty_cache()
    # 触发垃圾回收，释放未使用的内存
    gc.collect()
    return images


def ip_sdxl_portrait():
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    images = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]

    faceid_embeds = []
    for image in images:
        image = cv2.imread("person.jpg")
        faces = app.get(image)
        faceid_embeds.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
    faceid_embeds = torch.cat(faceid_embeds, dim=1)

    ip_ckpt = "/nvme0n1-disk/ipchange/IP-Adapter/IP-Adapter/hub_sdxl_models/IP-Adapter-FaceID/ip-adapter-faceid-portrait_sdxl.bin"
    device = "cuda"

    sdxlPipe = CustomInpaintPipeline()

    pipe = sdxlPipe.pipe

    # load ip-adapter
    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device, num_tokens=16, n_cond=5)

    # generate image
    prompt = "photo of a woman in red dress in a garden"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=4, width=1024,
        height=1024, num_inference_steps=30, seed=2023
    )
    return images
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import io
import numpy as np
from typing import Optional
import asyncio,sys,os

app = FastAPI()
async def restart_program():
    """Restarts the current program."""
    logger.info("Restarting program...")
    # await asyncio.sleep(5)  # 延迟5秒，确保响应已发送
    python_executable = sys.executable  # 使用当前 Python 的可执行文件路径
    os.execv(python_executable, [python_executable] + sys.argv)

@app.get("/re")
async def inpaint(
    tasks: BackgroundTasks
):
    logger.info('re start open_pose es ----------------')
    # 重启示范内存
    tasks.add_task(restart_program)

def try_get_face(img_path):
    re_images = []  # 用于存储所有裁剪的头部图像
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    try:
        app.prepare(ctx_id=0)
        # 读取图像
        image = cv2.imread(img_path)
        # 人脸检测
        faces = app.get(image)

        if faces:
            for face_t in faces:  # 遍历所有检测到的脸部
                # 获取头部矩形边界（通过关键点）
                x1, y1 = np.min(face_t.kps, axis=0).astype(int)
                x2, y2 = np.max(face_t.kps, axis=0).astype(int)

                # 计算脸部宽度和高度
                face_width = x2 - x1
                face_height = y2 - y1

                # 动态计算 padding（以确保包含头发和下巴）
                padding_x = int(face_width * 1.5)  # 横向增加 60% 的 padding
                padding_y = int(face_height * 1.5)  # 纵向增加 60% 的 padding

                # 更新裁剪区域，加上 padding
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(image.shape[1], x2 + padding_x)
                y2 = min(image.shape[0], y2 + padding_y)

                # 裁剪图像，保持裁剪后的原始分辨率
                head_image = image[y1:y2, x1:x2]

                # 转换为 RGB 格式并转为 PIL 图像
                head_image_rgb = cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB)
                head_pil_image = Image.fromarray(head_image_rgb)

                # 将裁剪后的图像添加到列表中
                re_images.append(head_pil_image)
        else:
            print("未检测到人脸")
    except Exception as e:
        # 打印完整的错误堆栈信息
        print("Error occurred:")
        traceback.print_exc()
        del app
    finally:
        # 释放 GPU 缓存
        torch.cuda.empty_cache()
        # 触发垃圾回收，释放未使用的内存
        gc.collect()
    return re_images

@app.post("/face")
async def inpaint(
    file_path: str = Form(...)
):
    logger.info(f'start find face -----{file_path}-----------')
    try:
        total_pic = try_get_face(file_path)
        # 创建一个 BytesIO 对象，作为 ZIP 文件的内存存储
        zip_io = io.BytesIO()
        # 创建 ZIP 文件
        with zipfile.ZipFile(zip_io, mode='w') as zip_file:
            for idx, img in enumerate(total_pic):
                img_io = io.BytesIO()
                # 将每张图片保存到 BytesIO 中
                img.save(img_io, format='PNG')
                img_io.seek(0)
                # 将图片添加到 ZIP 文件中，名称为 image_0.png, image_1.png 等
                zip_file.writestr(f'image_{idx}.png', img_io.getvalue())
        # 设置 ZIP 文件的起始位置
        zip_io.seek(0)
        re_it = StreamingResponse(zip_io, media_type="application/zip",
                          headers={"Content-Disposition": "attachment; filename=images.zip"})
        logger.info('suc gen img ------')
        return re_it
    except Exception as e:
        # 打印完整的错误堆栈信息
        print("Error occurred:")
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/inpaint")
async def inpaint(
    file_path: str = Form(...),  # 从表单接收文件路径
    prompt: str = Form(...),  # 从表单接收 prompt 字符串
    seed: int = Form(...),
    room_id: Optional[str] = Form(None),
    gen_type: Optional[str] = Form(None),
    only_file_path: Optional[str] = Form(None)
):
    try:
        logger.info(f'text gen en_prompt {prompt} -----{file_path}-----------{gen_type}--room_id {room_id}')
        total_pic = []
        if gen_type is None or gen_type == '':
            images_v2 = ip_sdxl_plus_v2(face_pic_path = file_path, prompt=prompt, seed=seed, only_face_pic_path= only_file_path, room_id=room_id)
            total_pic.extend(images_v2)
            # images_v0 = ip_sdxl(face_pic_path = file_path, prompt=en_prompt)
            # total_pic.extend(images_v0)
            images_v1 = ip_sdxl_plus_v1(face_pic_path = file_path, prompt=prompt, seed=seed, only_face_pic_path= only_file_path, room_id=room_id)
            total_pic.extend(images_v1)
        else:
            # flux
            images_flux = gen_img_and_swap_face(prompt, room_id=room_id)
            if only_file_path is not None and only_file_path != '':
                face_pic_path = only_file_path
            else:
                face_pic_path = file_path
            print(f'{face_pic_path}')
            temp_save_path = os.path.dirname(face_pic_path)
            temp_save_file = temp_save_path + '/' +str(uuid.uuid4()) + '.png'
            org_faces_f = [temp_save_file]
            to_faces_f = [face_pic_path]
            images_flux.save(temp_save_file)
            replace_result_img_5007 = req_replace_face(pic_b=temp_save_file, source_path_list=org_faces_f,
                                                       target_path_list=to_faces_f, port=5007)
            if replace_result_img_5007 is not None:
                total_pic.append(replace_result_img_5007)
            else:
                total_pic.append(images_flux)
        # 创建一个 BytesIO 对象，作为 ZIP 文件的内存存储
        zip_io = io.BytesIO()

        # 创建 ZIP 文件
        with zipfile.ZipFile(zip_io, mode='w') as zip_file:
            for idx, img in enumerate(total_pic):
                img_io = io.BytesIO()
                # 将每张图片保存到 BytesIO 中
                img.save(img_io, format='PNG')
                img_io.seek(0)
                # 将图片添加到 ZIP 文件中，名称为 image_0.png, image_1.png 等
                zip_file.writestr(f'image_{idx}.png', img_io.getvalue())

        # 设置 ZIP 文件的起始位置
        zip_io.seek(0)
        re_it = StreamingResponse(zip_io, media_type="application/zip",
                          headers={"Content-Disposition": "attachment; filename=images.zip"})
        logger.info('suc gen img ------')
        return re_it
    except Exception as e:
        # 打印完整的错误堆栈信息
        print("Error occurred:")
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1060)

# lustifySDXLNSFWSFW_v30.safetensors
# Camera type tags: "shot on Polaroid SX-70", "shot on Kodak Funsaver", "shot on GoPro Hero", "shot on Canon EOS 5D", "shot on Leica T".
# Photography styles: "analog photo", "glamour photography", "street fashion photography", "candid photo", "amateur photo".
# Lighting types: "cinematic lighting", "neon lighting", "soft lighting", "dramatic lighting", "low key lighting", "bright flash photography", "warm golden hour lighting", "radiant god rays".
# Film types: "Ilford HP5 Plus", "Lomochrome color film", "Fujicolor Pro".
# Photographers: Alessio Albi, Martin Schoeller, Miles Aldridge, Oleg Oprisco, Tim Walker.
# Others: "film grain", "bokeh", "dreamy haze", "technicolor", "underexposed", "low quality", "lowres"
# Creating Photorealistic Images With AI.pdf


#acornIsBoningXL_xlV2.safetensors
# As with my previous 1.5 versions, I made a randomizer prompt that takes advantage of dynamic prompts. You can set the seed to randomize and run off batches. Go back to tweak those than stand out to you. Enjoy.
# highest quality photograph of beautiful naked woman {(vaginal sex:1.5) with man, erection | (from above, BLOWJOB, big cock) | (manually masturbating:1.3) with (blue glass dildo insertion:1.3) | (kneeling), (receives cum in open mouth from man's cock:1.4) | (squatting, peeing piss:1.4) | (licking penis, cum:1.4) | (giving deep throat blowjob:1.4) | (having sex with another woman:1.4), (manual dildo inserted:1.5) | (POV BLOWJOB:1.4) | (having anal sex:1.4) | (kneeling masturbation, touching own pussy:1.4)} {in the bathroom | in hot tub | in the kitchen | by the pool | in bedroom | on rooftop in the city}, {ginger | blonde | brunette} {long | medium | short} hair, {petite | skinny | chubby} body, {(large breasts:1.3) | (medium breasts:1.3) | (small breasts:1.3) | (huge breasts:1.3)}, (intricately detailed skin), (perfect lips), 18 years old, high contrast
# Negative: (watermark:1.5), (anime, eye contact, misaligned:1.4), (missing limbs:1.5), (blurry), (deformed iris), (deformed legs) (extra feet, extra hands), (lipstick), (two dicks)
# DPM++ 2M SDE Heun Exponential is my preferred sampler but many work just fine.
# I prefer to use 40 steps or more, but you can get good results at lower steps as well, at least down to 20.
# CFGs from 3 to 9 all work pretty well.

#STOIQOAfroditeFLUXXL_XL31.safetensors


