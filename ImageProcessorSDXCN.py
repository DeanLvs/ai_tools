from PIL import Image
import os
import hashlib
import torch
import gc
from CustCoun import encode_prompt_with_cache
from BookYesCommon import GenContextParam, GenSavePathParam
from diffusers import (StableDiffusionXLControlNetInpaintPipeline, EulerAncestralDiscreteScheduler,
                       DEISMultistepScheduler, LMSDiscreteScheduler, HeunDiscreteScheduler, RePaintScheduler,
                       StableDiffusionXLInpaintPipeline, ControlNetModel, UNet2DConditionModel, AutoencoderKL,
                       EulerDiscreteScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPTextModelWithProjection
from safetensors.torch import load_file as load_safetensors
from transUnet import get_new_key
from transVae import transform_line
from ccxx import convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint
from text_encoder_2_conver import convert_keys
from book_yes_logger_config import logger
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
            self.cache = {}  # 内存缓存字典
            self._initialized = True
            # —— 新增：加载状态标志
            self.is_loaded = False
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_path = model_path
            self.controlnet_list = controlnet_list_t #,'depth','canny','key_points','depth',,'seg'
            self.use_out_test = use_out_test
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
            # weights_path = "/nvme0n1-disk/civitai-downloader/realvisxlV50_v50Bakedvae.safetensors"
            weights_path = "/nvme0n1-disk/civitai-downloader/juggernautXL_v9rdphoto2Inpaint.safetensors"
            self.load_pretrained_weights(weights_path)
            # adeptsdxlV40_v40.safetensors 10.9 acornIsBoningXL_xlV2.safetensors 黑了  juggernautxlJuggxi_juggernautxl.safetensors 10.9
            #  epicrealismXL_v8KISS-inpainting.safetensors nfsw
            # AsianGirl.safetensors为了获得更好的效果，建议使用：ADETAILER mediapipe face mesh。 设置：采样 Euler-a 或 DPM Karras、HiresFix 开启、R-ESRGAN 4x+、去噪 0.4、高档 2x、采样步长 35+、分辨率 512x768、高档 4x Ultra Sharp 下一段。


            # envyStarlightBody_v10.safetensors 素描人体解刨图 倒不进去
            # johnslegersart_nudes.safetensors 裸体 每张都都有标签字幕  没效果 再试试
            # nudity_v02_i1300.safetensors  0.9 1.0不行感觉

            # finenude_v0_2a.safetensors 裸体 1.0 0.6 不行
            # NudeXL.safetensors 不行，没有效果 皮肤不漏了

            # NaturalBodyV2.0.safetensors 真实裸体 0.9 1.0 realistic nude completely nude 真了一些 比例 有改善
            # biggunsxl_v11.safetensors 不行吖
            # 0.8 BT_Detailed_Female_Belly_XL.safetensors 腹部线条 有效果 但是没改变比例
            # SDXL_DatAss_v1.safetensors 1.0不行 改变强度试试
            # Realistic_Pussy_Xl-000010.safetensors
            # Pussy_Cameltoe_XL_v2.safetensors 逼真pussy
            # Pussy_Cameltoe_XL_v2.safetensors cameltoe_body - 全身图像；最好配有肖像  cameltoe_hips - 仅限臀部；最适合方形和横向 cameltoe_pussy - 外阴特写；最好是方形和横向
            # self.pipe.load_ip_adapter("InvokeAI/ip_adapter_sdxl_image_encoder")
            # self.pipe.load_ip_adapter("InvokeAI/ip_adapter_sdxl")
            # self.load_lora_weights(lora_path="/nvme0n1-disk/civitai-downloader/lora/biggunsxl_v11.safetensors", lora_scale=0.5, adapter_name="bigsxiiiixxl")
            # self.load_lora_weights(lora_path="/nvme0n1-disk/civitai-downloader/lora/SDXL_DatAss_v1.safetensors", lora_scale=0.7, adapter_name="datass")
            # self.load_lora_weights(lora_path="/nvme0n1-disk/civitai-downloader/lora/Realistic_Pussy_Xl-000010.safetensors", lora_scale=0.8, adapter_name="pppussy")
            self.load_lora_weights_base(lora_path="/nvme0n1-disk/civitai-downloader/lora/nudify_xl_lite.safetensors", lora_scale=0.6, adapter_name="binu")
        logger.info('suc lora')

    def release_resources(self):
        """
        彻底释放所有模型和缓存，并重置加载状态。
        """
        if getattr(self, "is_loaded", False):
            # —— 1. 先把整个 pipeline 和子模块都搬回 CPU
            try:
                self.pipe.to("cpu")
                logger.info("Pipeline moved to CPU.")
            except Exception as e:
                logger.warning(f"Moving pipeline to CPU failed: {e}")

            # —— 2. 如果有 LoRA 层，先卸载
            try:
                self.pipe.unload_lora_weights()
                logger.info("Unloaded LoRA weights.")
            except Exception as e:
                logger.warning(f"Unloading LoRA failed: {e}")

            # —— 3. 删除 pipeline 内部所有可能的子模块引用
            for sub_attr in ["unet", "vae", "controlnet", "scheduler", "text_encoder", "text_encoder_2"]:
                if hasattr(self.pipe, sub_attr):
                    try:
                        delattr(self.pipe, sub_attr)
                        logger.info(f"Deleted pipe.{sub_attr}")
                    except Exception:
                        pass

            # —— 4. 删除整个 pipeline 引用
            try:
                del self.pipe
            except Exception:
                pass

            # —— 5. 再把 tokenizer、encoder 等引用也删掉
            for attr in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]:
                if hasattr(self, attr):
                    try:
                        delattr(self, attr)
                    except Exception:
                        pass

            # —— 6. 强制回收
            torch.cuda.empty_cache()
            # PyTorch 2.1+ 可选：torch.cuda.ipc_collect()
            gc.collect()

            # —— 7. 标记已释放
            self.is_loaded = False
            logger.info("CustomInpaintPipeline: resources released, is_loaded=False")

        # 最后重置单例
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
        scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), solver_order=2, algorithm_type = "sde-dpmsolver++", use_karras_sigmas=True)
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), solver_order = 3, use_karras_sigmas=True)
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
        # 手动加载 VAE
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
            pipe = StableDiffusionXLInpaintPipeline(
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
        pipe = StableDiffusionXLControlNetInpaintPipeline(
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

    def load_lora_weights_base(self, lora_path=None, adapter_name=None, lora_scale=0.7):
        logger.info("Loaded LoRA layers:")
        self.pipe.load_lora_weights(lora_path, adapter_name = adapter_name, force_download=True)
        self.pipe.fuse_lora(lora_scale=lora_scale)
        # self.pipe.set_adapters(['test'], adapter_weights=[1.0])
        # logger.info(f" - {lora_load_state_dict}")
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
                # aesthetic_score=6.0,
                # negative_aesthetic_score=2.5,
                callback_steps=1  # 调用回调的步数
            ).images[0]
        # 释放 GPU 内存
        if 'init_image' in locals():
            del init_image
        else:
            logger.info("init_image does not exist.")
        if 'mask_image' in locals():
            del mask_image
        else:
            logger.info("mask_image does not exist.")
        torch.cuda.empty_cache()
        gc.collect()
        return result

    def genIt(self, g_c_param: GenContextParam):
        gen_it_prompt = ''
        try:
            logger.info(f"gen use prompt: {g_c_param}")
            # ip_adapter_image = Image.open("IMG_5132.JPG").convert("RGB")
            if self.use_out_test:
                g_c_param.prompt = None
                g_c_param.prompt_2 = None
                g_c_param.reverse_prompt = None
                g_c_param.reverse_prompt_2 = None
            with torch.no_grad():
                # 获取编码结果（独立存储并缓存）
                # 从 encode_and_cache 获取所有的 prompt_embeds 和 pooled_prompt_embeds
                # final_prompt_embeds, final_negative_prompt_embeds, final_pooled_prompt_embeds, final_pooled_negative_prompt_embeds = processor.encode_and_cache_it(prompt, prompt_2, reverse_prompt, reverse_prompt_2)
                if len(self.controlnet_list) == 0:
                    logger.info(
                        f'run ----------------------------- use Inpaint Pipeline {g_c_param}')
                    resultTemp = self.pipe(
                        image=g_c_param.next_genImage,
                        prompt=g_c_param.prompt,
                        prompt_2=g_c_param.prompt_2,
                        negative_prompt=g_c_param.reverse_prompt,
                        negative_prompt_2=g_c_param.reverse_prompt_2,
                        prompt_embeds=g_c_param.prompt_embeds,
                        pooled_prompt_embeds=g_c_param.pooled_prompt_embeds,
                        negative_prompt_embeds=g_c_param.negative_prompt_embeds,  # 负向提示词嵌入
                        negative_pooled_prompt_embeds=g_c_param.negative_pooled_prompt_embeds,  # 负向 pooled embeddings
                        mask_image=g_c_param.big_mask,
                        # ip_adapter_image=ip_adapter_image,
                        num_inference_steps=g_c_param.num_inference_steps,
                        guidance_scale=g_c_param.guidance_scale,
                        callback=g_c_param.func_call_back,
                        strength=g_c_param.strength,
                        # aesthetic_score=6.0,
                        # negative_aesthetic_score=2.5,
                        # padding_mask_crop=32,
                        callback_steps=1  # 调用回调的步数
                    ).images[0]
                else:
                    control_image_list_t = []
                    controlnet_conditioning_scale_t = []
                    logger.info(f'use {self.controlnet_list}')
                    for cont in self.controlnet_list:
                        if 'key_points' == cont:
                            control_image_list_t.append(g_c_param.control_img_list[0])
                            controlnet_conditioning_scale_t.append(g_c_param.control_float_array[0])
                        elif 'depth' == cont:
                            if g_c_param.control_img_list[1]:
                                logger.info(f'use depth line pic---------')
                                control_image_list_t.append(g_c_param.control_img_list[1])
                                controlnet_conditioning_scale_t.append(g_c_param.control_float_array[1])
                            else:
                                logger.info(f'use def none depth pic pic---------')
                                t_img_width, t_img_height = g_c_param.next_genImage.size
                                control_image_list_t.append(Image.new('L', (t_img_width, t_img_height), 0))
                                controlnet_conditioning_scale_t.append(0)
                        elif 'seg' == cont:
                            control_image_list_t.append(g_c_param.control_img_list[2])
                            controlnet_conditioning_scale_t.append(g_c_param.control_float_array[2])
                        elif 'canny' == cont:
                            if g_c_param.control_img_list[3]:
                                logger.info(f'use paint line pic---------')
                                control_image_list_t.append(g_c_param.control_img_list[3])
                                controlnet_conditioning_scale_t.append(g_c_param.control_float_array[3])
                            else:
                                logger.info(f'use def none paint line pic---------')
                                t_img_width, t_img_height = g_c_param.next_genImage.size
                                control_image_list_t.append(Image.new('L', (t_img_width, t_img_height), 0))
                                controlnet_conditioning_scale_t.append(0)

                    if len(control_image_list_t) == 1:
                        control_image_list_f = control_image_list_t[0]
                        controlnet_conditioning_scale_f = controlnet_conditioning_scale_t[0]
                    else:
                        control_image_list_f = control_image_list_t
                        controlnet_conditioning_scale_f = controlnet_conditioning_scale_t
                    logger.info(
                        f'run ----------------------------- use Controlnet Pipeline {g_c_param}')
                    resultTemp = self.pipe(
                        image=g_c_param.next_genImage,
                        prompt=g_c_param.prompt,
                        prompt_2=g_c_param.prompt_2,
                        negative_prompt=g_c_param.reverse_prompt,
                        negative_prompt_2=g_c_param.reverse_prompt_2,
                        # ip_adapter_image=ip_adapter_image,
                        prompt_embeds=g_c_param.prompt_embeds,
                        pooled_prompt_embeds=g_c_param.pooled_prompt_embeds,
                        negative_prompt_embeds=g_c_param.negative_prompt_embeds,  # 负向提示词嵌入
                        negative_pooled_prompt_embeds=g_c_param.negative_pooled_prompt_embeds,  # 负向 pooled embeddings
                        mask_image=g_c_param.big_mask,
                        num_inference_steps=g_c_param.num_inference_steps,
                        guidance_scale=g_c_param.guidance_scale,
                        callback=g_c_param.func_call_back,
                        strength=g_c_param.strength,
                        control_image=control_image_list_f,
                        controlnet_conditioning_scale=controlnet_conditioning_scale_f,
                        # padding_mask_crop=32,
                        callback_steps=1  # 调用回调的步数
                    ).images[0]
        finally:
            # 清理已使用的张量
            # 在使用完后显式删除变量以释放内存
            # 释放 GPU 内存
            torch.cuda.empty_cache()
            gc.collect()
        return resultTemp, gen_it_prompt


