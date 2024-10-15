from PIL import Image
import os
import torch
import gc
import SingletonOpenposeDetector
from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from ControlNetPlus.pipeline.pipeline_controlnet_union_inpaint_sd_xl import StableDiffusionXLControlNetUnionInpaintPipeline
from ControlNetPlus.models.controlnet_union import ControlNetModel_Union
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPTextModelWithProjection
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

    def __init__(self, model_path="stable-diffusion-xl-1.0-inpainting-0.1"):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_path = model_path
            self.pipe = self._initialize_pipe()
            weights_path = "/mnt/sessd/civitai-downloader/juggernautXL_v9rdphoto2Inpaint.safetensors"
            self.load_pretrained_weights(weights_path)
            # Pussy_Cameltoe_XL_v2.safetensors 0.7
            # biggunsxl_v11.safetensors 07
            # 717006  BustyBabe 大乳房
            # 481415 3D 事情
            # 716470 jdmlaceteddy1
            # 712510 裸体女 比较小数据
            # 712827 虚拟女 Rebecca v1.0
            # 518563 虚拟女 Candie v1.0
            # 464765 Veronica v1.0
            # 464780 Elissa
            # 448760 katya
            # 448332 tori
            # 364491 joey
            # 347514 lacey
            # 336121 malena
            # 333727 dawn
            # 699905 a beautiful photo of as1an, 1girl, detailed, 8k, masterpiece, realistic, eyelashes
            # 451519 高跟鞋
            # "/mnt/sessd/civitai-downloader/lora/Nipples XL" nudify_xl.safetensors https://civitai.com/api/download/models/381060?type=Model&format=SafeTensor 381060 0.7 bools 360150 0.7 psy
            # self.load_lora_weights(lora_path="/mnt/sessd/civitai-downloader/lora/NaturalBodyV2.0.safetensors", lora_scale= 0.5, adapter_name= "naturalbbb")
            #NudeXL.safetensors 0.3
            # ptgs.safetensors 0.1     Breasts-05.safetensors 乳头
            # self.load_lora_weights(lora_path="/mnt/sessd/civitai-downloader/lora/biggunsxl_v11.safetensors",
            #                        lora_scale=0.6, adapter_name="bigsxiiiixxl")
            # self.load_lora_weights(lora_path="/mnt/sessd/civitai-downloader/lora/Breasts-05.safetensors",
            #                        lora_scale=0.7, adapter_name="bigsxlxssxsd")
            # self.load_lora_weights(lora_path="/mnt/sessd/civitai-downloader/lora/Nipples XL - All in One.safetensors", lora_scale=0.5, adapter_name="nudifyitw")
            # self.load_lora_weights(lora_path="/mnt/sessd/civitai-downloader/lora/NudeXL.safetensors",
            #                        lora_scale=0.3, adapter_name="bigsxlxxyyaaa")

            print('suc lora')
            # '''
            # -rw-r--r-- 1 root root  6938072392 Aug  1 20:31  animaginexl_v31Inpainting.safetensors  动漫 男的
            # -rw-r--r-- 1 root root  6938070578 Aug  1 12:28  epicrealismXL_v8KISS-inpainting.safetensors 不错 尺度 最大
            # -rw-r--r-- 1 root root  6938071448 Aug  1 20:22  haveallsdxlInpaint_v10.safetensors 一般 比例也不对，非常一般
            # -rw-r--r-- 1 root root  7105380384 Aug  4 02:51  inpaintSDXLPony_inpaintPony.safetensors 变花-----
            # -rw-r--r-- 1 root root  6938072930 Aug  1 22:08  inpaintSDXLPony_juggerInpaintV8.safetensors  也不错
            #
            # -rw-r--r-- 1 root root  6938072736 Aug  7 00:11  juggernautXL_v9rdphoto2Inpaint.safetensors 也不错  尺度不大
            # -rw-r--r-- 1 root root  6938072258 Aug  3 18:55  juggernautXL_versionXInpaint.safetensors  也不错  和 juggernautXL_v9rdphoto2Inpaint.safetensors 一样 但是尺度好像大
            # -rw-r--r-- 1 root root  6938070722 Aug  4 10:35  lustifySDXLNSFW_v10-inpainting.safetensors   不行 分辨率 太低  但是效果 还可以，   指导说要处理，算了--------
            # -rw-r--r-- 1 root root  6938070650 Aug  4 02:18  realvisxlv40_-inpainting.safetensors      这个可以，分辨率不低 似乎尺度不大  比 juggernautXL_v9rdphoto2Inpaint.safetensors 好
            # -rw-r--r-- 1 root root  6938069482 Aug  6 02:03  realvisxlV40_v30InpaintBakedvae.safetensors   这个可以，分辨率不低 似乎尺度不大
            # -rw-r--r-- 1 root root  6938069482 Aug  1 20:30  realismFromHadesXL_v60XLInpainting.safetensors  不行 不出图
            # -rw-r--r-- 1 root root  7105380384 Aug  6 02:05  sevenof9PonyRealMix_inpaintPony.safetensors 变花-----
            # -rw-r--r-- 1 root root  6938069944 Aug  1 20:29  imaginariumInpaint_v10.safetensors 也不错
            # -rw-r--r-- 1 root root  6938040682 Aug  3 16:55  talmendoxlSDXL_v11Beta.safetensors 不出图
            # '''
            # lustifySDXLNSFW_v10-inpainting.safetensors 清晰度下降
            # realvisxlV40_v30InpaintBakedvae.safetensors  试一下 worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch ,open mouth
            # jibMixRealisticXL_v140crystalclarity.safetensors 可以 也不要猜疑
            # inpaintSDXLPony_inpaintPony.safetensors 一会试一下 juggernautXL_v9rdphoto2Inpaint.safetensors
            # inpaintSDXLPony_juggerInpaintV8.safetensors 最好目前
            # realismFromHadesXL_v60XLInpainting.safetensors 不行

            # animaginexl_v31Inpainting.safetensors 动漫
            # imaginariumInpaint_v10.safetensors 不行
            # haveallsdxlInpaint_v10.safetensors 不行
            #  realisticVisionV60B1_v51HyperInpaintVAE.safetensors
            # realismFromHadesXL_v60XLInpainting.safetensors 正经模型


            # epicrealismXL_v8KISS-inpainting.safetensors 还行
            # sevenof9PonyRealMix_inpaintPony.safetensors 清晰度蓝 ，改下采样器 40-60 以上 steps   7 看 说明1 还是不行


            # pornmasterPro_v8-f16-inpainting.safetensors sd1.5
            # pornmasterPro_v8-inpainting-b.safetensors sd1.5
            # clarity_V3-inpainting.safetensors sd1.5
            # juggernautXL_v9rdphoto2Inpaint.safetensors 效果不错
            # juggernautXL_versionXInpaint.safetensors     一会试试
            # animaginexl_v31Inpainting.safetensors

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

    def _initialize_pipe(self):
        # 手动加载调度器
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), use_karras_sigmas=True)
        # scheduler = EulerDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), solver_order=3)

        # juggernautXL_v9rdphoto2Inpaint.safetensors 使用
        scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), use_karras_sigmas=True)

        # scheduler = DDIMScheduler.from_pretrained(os.path.join(self.model_path, "text_encoder"))

        # lustifySDXLNSFW_v10-inpainting.safetensors 使用
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
        # scheduler.use_karras_sigmas = True
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

        controlnet_model = ControlNetModel_Union.from_pretrained("xinsir/controlnet-union-sdxl-1.0",
                                                                 torch_dtype=torch.float16, use_safetensors=True)
        pipe = StableDiffusionXLControlNetUnionInpaintPipeline(
            controlnet=controlnet_model,
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
        text_encoder_keys = self.pipe.text_encoder.state_dict().keys()
        loaded_text_encoder = modified_state_dict
        if 't1' in load_part_list:
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
        modified_state_dict = convert_keys(state_dict)
        for key in modified_state_dict.keys():
            if key in text_2_model_dict:
                if modified_state_dict[key].shape == text_2_model_dict[key].shape:
                    loaded_text_encoder[key] = modified_state_dict[key]
                else:
                    print(f"keys shape: {key} checkpoint shape is {modified_state_dict[key].shape} but model shape is {text_2_model_dict[key].shape}\n")
        if 't2' in load_part_list:
            text_encoder_load_info = self.pipe.text_encoder_2.load_state_dict(loaded_text_encoder, strict=False)
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
        self.pipe.load_lora_weights(lora_path, adapter_name = adapter_name, force_download=True)
        self.pipe.fuse_lora(lora_scale=lora_scale)
        # self.pipe.set_adapters(['test'], adapter_weights=[1.0])

        # print(f" - {lora_load_state_dict}")
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

def load_pretrained_weights(weights_path, unet, vae, text_encoder, text_encoder_2):
    if not weights_path or weights_path == '':
        print(f"no check point config")
        return
    # 加载预训练权重
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"权重文件未找到: {weights_path}")

    state_dict = load_safetensors(weights_path)
    # 替换 key
    unet_state_dict = convert_ldm_unet_checkpoint(state_dict)
    unet_model_dict = unet.state_dict()
    loaded_unet = {}
    # 仅加载形状匹配的键
    for key in unet_state_dict.keys():
        if key in unet_model_dict:
            if unet_state_dict[key].shape == unet_model_dict[key].shape:
                loaded_unet[key] = unet_state_dict[key]
            else:
                print(f"this {key} len shape {unet_state_dict[key].shape} and sa shape {unet_model_dict[key].shape}")

    print(f"this {weights_path} len is {len(state_dict.keys())}")
    # 加载 UNet 权重
    print("Loading UNet weights...")
    unet_keys = unet_model_dict.keys()
    unet_load_info = unet.load_state_dict(loaded_unet, strict=False)
    print(f"UNet total keys: {len(unet_keys)}, loaded keys: {len(loaded_unet)}, missing keys: {len(unet_load_info.missing_keys)}")
    print(f"Missing UNet keys: {unet_load_info.missing_keys}\n")

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
    vae_model_dict = vae.state_dict()  # 获取完整的 VAE 模型状态字典
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

    vae_load_info = vae.load_state_dict(loaded_vae, strict=False)
    print(
        f"VAE total keys: {len(vae_model_dict.keys())}, loaded keys: {len(loaded_vae)}, missing keys: {len(vae_load_info.missing_keys)}")
    print(f"Missing VAE keys: {vae_load_info.missing_keys}\n")

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
    text_encoder_keys = text_encoder.state_dict().keys()
    loaded_text_encoder = modified_state_dict
    text_encoder_load_info = text_encoder.load_state_dict(loaded_text_encoder, strict=False)

    print(f"Text encoder total keys: {len(text_encoder_keys)}, loaded keys: {len(loaded_text_encoder)}, missing keys: {len(text_encoder_load_info.missing_keys)}")
    print(f"Missing text encoder keys: {text_encoder_load_info.missing_keys}\n")

    # 加载文本编码器权重
    print("Loading text encoder 2 weights...")
    # 移除键名中的 'conditioner.embedders.0.transformer.' 前缀
    print("Loading text encoder 2 weights...")
    text_encoder_keys = text_encoder_2.state_dict().keys()

    loaded_text_encoder = {}

    text_2_model_dict = text_encoder_2.state_dict()
    # 仅加载形状匹配的键
    modified_state_dict = convert_keys(state_dict)
    for key in modified_state_dict.keys():
        if key in text_2_model_dict:
            if modified_state_dict[key].shape == text_2_model_dict[key].shape:
                loaded_text_encoder[key] = modified_state_dict[key]
            else:
                print(f"keys shape: {key} checkpoint shape is {modified_state_dict[key].shape} but model shape is {text_2_model_dict[key].shape}\n")

    text_encoder_load_info = text_encoder_2.load_state_dict(loaded_text_encoder, strict=False)
    print(
        f"Text encoder 2 total keys: {len(text_encoder_keys)}, loaded keys: {len(loaded_text_encoder)}, missing keys: {len(text_encoder_load_info.missing_keys)}")
    print(f"Missing text encoder 2 keys: {text_encoder_load_info.missing_keys}\n")


if __name__ == '__main__':
    # 使用示例
    pipeline = CustomInpaintPipeline()
    # pipeline.load_lora_weights("/mnt/sessd/civitai-downloader/nudify_xl.safetensors")
    # 使用示例
    init_image_path = "/mnt/sessd/ai_tools/static/uploads/3fc3cd52ced292773a155b9e9f90933b.png"  # 替换为实际的初始图像路径
    mask_image_path = "/mnt/sessd/ai_tools/static/uploads/filled_use_mask_image_pil_3fc3cd52ced292773a155b9e9f90933b.png"  # 替换为实际的遮罩图像路径

    # 加载图像
    init_image = Image.open(init_image_path).convert("RGB")
    mask_image = Image.open(mask_image_path).convert("L")

    # 使用 OpenPose 检测器生成骨架图像
    # openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    # control_image = openpose(init_image)
    # control_image = control_image.resize(init_image.size)
    # control_image.save('xxxxxxxxpose.png')
    # original_width, original_height = init_image.size
    result_image = pipeline.generate_image(init_image=init_image, mask_image=mask_image, control_image=None)

    # 高分辨率调整
    # upscale_factor = 1.5  # 放大比例，范围1.4-1.5
    # highres_image = result_image.resize((int(original_width * upscale_factor), int(original_height * upscale_factor)), Image.LANCZOS)
    #
    #
    # result_image = result_image.resize((original_width, original_height), Image.LANCZOS)
    result_image.save("output555123212.png")
