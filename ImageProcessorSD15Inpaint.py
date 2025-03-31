from PIL import Image
from closeMaskIt import save_binary_clothing_mask
from PIL import Image, ImageOps, ImageFilter
import os
import torch
import gc,json
from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler,ControlNetModel, UNet2DConditionModel, AutoencoderKL, PNDMScheduler, DPMSolverMultistepScheduler
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
    def __init__(self, model_path="/Users/dean/sd15/stable-diffusion-v1-5-inpainting",
                 controlnet_list_t = [], use_out_test = False):#,'depth'
        if not hasattr(self, "_initialized"):
            self.cache = {}  # 内存缓存字典
            self._initialized = True
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_path = model_path
            self.use_out_test = use_out_test
            self.controlnet_list = controlnet_list_t #,'depth','canny','key_points','depth',,'seg'
            self.text_device = self.device
            # 手动加载第一个文本编码器
            text_encoder = CLIPTextModel.from_pretrained(os.path.join(self.model_path, "text_encoder")).to(self.text_device) #, torch_dtype=torch.float16
            # 手动加载第一个分词器
            tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer"))
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer
            self._print_component_load_info("Text Encoder 1", text_encoder)
            self.pipe = self._initialize_pipe()
            # weights_path = "/Users/dean/sd15/pornmasterProFULLV4_fullV4-inpainting.safetensors"
            weights_path = "/Users/dean/sd15/pornmasterProV8_v8-inpainting.safetensors"
            self.load_model_weights(weights_path)
        print('suc lora')
    def _initialize_pipe(self):
        # 手动加载调度器
        # scheduler = EulerAncestralDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"),use_karras_sigmas=True)

        # scheduler = PNDMScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"),  use_karras_sigmas=True)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), solver_order=2, algorithm_type="dpmsolver++", use_karras_sigmas=True)
        # scheduler = PNDMScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"), use_karras_sigmas=True)
        vae = AutoencoderKL.from_pretrained(os.path.join(self.model_path, "vae")).to(self.device) # , torch_dtype=torch.float16
        # 手动加载 UNet
        unet = UNet2DConditionModel.from_pretrained(os.path.join(self.model_path, "unet")).to(self.device) # , torch_dtype=torch.float16
        # # 将各组件的键写入文件
        self._write_keys_to_file("vae_keys.txt", vae.state_dict().keys())
        self._write_keys_to_file("unet_keys.txt", unet.state_dict().keys())
        self._print_component_load_info("UNet", unet)
        self._print_component_load_info("VAE", vae)
        print(f"total keys: {len(unet.state_dict().keys()) + len(vae.state_dict().keys())} "
              f"text_encoder keys:  {len(self.text_encoder.state_dict().keys())} ")
        if self.use_out_test:
            to_text_encoder = None
            to_tokenizer = None

        else:
            to_text_encoder = self.text_encoder
            to_tokenizer = self.tokenizer

        if len(self.controlnet_list) == 0:
            print(f'init ----------------------------- use Normal Pipeline '
                  f'{type(to_text_encoder)} {type(to_tokenizer)} ')
            pipe = StableDiffusionInpaintPipeline(
                vae=vae,
                text_encoder=to_text_encoder,
                tokenizer=to_tokenizer,
                unet=unet,
                scheduler=scheduler,
                feature_extractor=None,
                safety_checker=None
            )
            pipe.to(self.device)
            return pipe
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
        pipe = StableDiffusionControlNetInpaintPipeline(
            controlnet=to_controlnet,
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
        self.pipe.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path, adapter_name=adapter_name)
        self.pipe.fuse_lora(lora_scale=lora_scale)
        print(f"loaded lora path: {lora_path}\n")

    def generate_image(self, image=None, mask_image = None, prompt="R/nsfw,nude,person,big and natural asses,big and natural boobs",
                       reverse_prompt="no deformed limbs,no disconnected limbs,unnaturally contorted position,no elongated waists, unrealistic features,overly exaggerated body parts,incorrect proportions, no multiple belly buttons",#deformed, bad anatomy, mutated, long neck, narrow Hips
                       num_inference_steps = 30, seed = 178, guidance_scale = 9.0,progress_callback=None,
                       strength=0.9, control_image=None, controlnet_conditioning_scale=None):
        torch.manual_seed(seed)
        result = self.pipe(
                image = image,
                mask_image = mask_image,
                prompt = prompt,
                negative_prompt= reverse_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback=progress_callback,
                strength=strength,
                control_image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                # padding_mask_crop=32,
                callback_steps=1  # 调用回调的步数
            ).images[0]
        torch.cuda.empty_cache()
        gc.collect()
        return result
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

    def genIt(self, prompt, prompt_2, reverse_prompt, reverse_prompt_2, genImage, big_mask,
              num_inference_steps, guidance_scale, progress_callback=None, strength=0.8,
              control_image=[None, None, None, None],
              controlnet_conditioning_scale=[1.0, 1.0, 1.0, 1.0],
              prompt_embeds=None, negative_prompt_embeds=None,
              pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None):
        gen_it_prompt = ''
        try:
            print(f"gen use prompt: {prompt} reverse_prompt: {reverse_prompt} "
                  f"num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}, "
                  f"strength: {strength} prompt_2: {prompt_2} reverse_prompt_2: {reverse_prompt_2} "
                  f"control_image: {control_image} controlnet_conditioning_scale_t: {controlnet_conditioning_scale}")
            # ip_adapter_image = Image.open("IMG_5132.JPG").convert("RGB")
            if self.use_out_test:
                prompt = None
                prompt_2 = None
                reverse_prompt = None
                reverse_prompt_2 = None
            with torch.no_grad():
                # 获取编码结果（独立存储并缓存）
                # 从 encode_and_cache 获取所有的 prompt_embeds 和 pooled_prompt_embeds
                # final_prompt_embeds, final_negative_prompt_embeds, final_pooled_prompt_embeds, final_pooled_negative_prompt_embeds = processor.encode_and_cache_it(prompt, prompt_2, reverse_prompt, reverse_prompt_2)
                if len(self.controlnet_list) == 0:
                    gen_it_prompt = (f'prompt:{prompt}#prompt_2:{prompt_2}#reverse_prompt:{reverse_prompt}'
                                     f'#reverse_prompt_2:{reverse_prompt_2}#strength:{strength}'
                                     f'#num_inference_steps:{num_inference_steps}#guidance_scale:{guidance_scale}'
                                     f'#prompt_embeds{type(prompt_embeds)}#pooled_prompt_embeds:{type(pooled_prompt_embeds)}'
                                     f'#negative_prompt_embeds:{type(negative_prompt_embeds)}'
                                     f'#negative_pooled_prompt_embeds:{type(negative_pooled_prompt_embeds)}')
                    print(
                        f'run ----------------------------- use Inpaint Pipeline R/FitNakedChicks {gen_it_prompt}')
                    resultTemp = self.pipe(
                        image=genImage,
                        prompt= "reddit, best quality,best aesthetic, R/FitNakedChicks, R/ass, R/boobs, R/pussy, nude",#prompt + ',' + prompt_2,
                        negative_prompt="",#reverse_prompt + ',' + reverse_prompt_2,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,  # 负向提示词嵌入
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,  # 负向 pooled embeddings
                        mask_image=big_mask,
                        # ip_adapter_image=ip_adapter_image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        callback=progress_callback,
                        strength=strength,
                        # padding_mask_crop=32,
                        callback_steps=1  # 调用回调的步数
                    ).images[0]
                else:
                    control_image_list_t = []
                    controlnet_conditioning_scale_t = []
                    print(f'use {self.controlnet_list}')
                    for cont in self.controlnet_list:
                        if 'key_points' == cont:
                            control_image_list_t.append(control_image[0])
                            controlnet_conditioning_scale_t.append(controlnet_conditioning_scale[0])
                        elif 'depth' == cont:
                            control_image_list_t.append(control_image[1])
                            controlnet_conditioning_scale_t.append(controlnet_conditioning_scale[1])
                        elif 'seg' == cont:
                            control_image_list_t.append(control_image[2])
                            controlnet_conditioning_scale_t.append(controlnet_conditioning_scale[2])
                        elif 'canny' == cont:
                            if control_image[3]:
                                print(f'use paint line pic---------')
                                control_image_list_t.append(control_image[3])
                                controlnet_conditioning_scale_t.append(controlnet_conditioning_scale[3])
                            else:
                                print(f'use def none paint line pic---------')
                                t_img_width, t_img_height = genImage.size
                                control_image_list_t.append(Image.new('L', (t_img_width, t_img_height), 0))
                                controlnet_conditioning_scale_t.append(0)

                    if len(control_image_list_t) == 1:
                        control_image_list_f = control_image_list_t[0]
                        controlnet_conditioning_scale_f = controlnet_conditioning_scale_t[0]
                    else:
                        control_image_list_f = control_image_list_t
                        controlnet_conditioning_scale_f = controlnet_conditioning_scale_t
                    gen_it_prompt = (f'prompt:{prompt}#prompt_2:{prompt_2}#reverse_prompt:{reverse_prompt}'
                                     f'#reverse_prompt_2:{reverse_prompt_2}#strength:{strength}'
                                     f'#num_inference_steps:{num_inference_steps}#guidance_scale:{guidance_scale}'
                                     f'#control_image_list_f:{control_image_list_f}#controlnet_conditioning_scale_f:{controlnet_conditioning_scale_f}'
                                     f'#prompt_embeds{type(prompt_embeds)}#pooled_prompt_embeds:{type(pooled_prompt_embeds)}'
                                     f'#negative_prompt_embeds:{type(negative_prompt_embeds)}'
                                     f'#negative_pooled_prompt_embeds:{type(negative_pooled_prompt_embeds)}')
                    print(
                        f'run ----------------------------- use Controlnet Pipeline {gen_it_prompt}')
                    resultTemp = self.pipe(
                        image=genImage,
                        prompt=prompt + ',' + prompt_2,
                        negative_prompt=reverse_prompt + ',' + reverse_prompt_2,
                        # ip_adapter_image=ip_adapter_image,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,  # 负向提示词嵌入
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,  # 负向 pooled embeddings
                        mask_image=big_mask,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        callback=progress_callback,
                        strength=strength,
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

def total_test():
    pippres = CustomInpaintPipeline()
    image = Image.open("filled_image_pil_IMG_5118.jpeg").convert("RGB")
    mask_image = Image.open("filled_use_mask_image_pil_IMG_5118.jpeg").convert("L")
    clear_org_i = Image.open("control_net_IMG_5118.jpeg").convert("RGB")
    depth_c = Image.open("nor_control_net_IMG_5118.jpeg").convert("RGB")
    ling_c = Image.open("line_auto_IMG_5118.jpeg").convert("RGB")
    control_image = clear_org_i # [clear_org_i, depth_c]
    controlnet_conditioning_scale = 1.0
    img = pippres.generate_image(image=image, mask_image=mask_image, control_image = control_image, controlnet_conditioning_scale = controlnet_conditioning_scale)
    img.save('511851185118.png')

    image = Image.open("IMG_5118.jpeg").convert("RGB")
    img = pippres.generate_image(image=image, mask_image=mask_image, control_image=control_image,
                                 controlnet_conditioning_scale=controlnet_conditioning_scale)
    img.save('511851185118-ORG.png')
def resize_with_padding(image, save_path, target_size=(512, 512)):
    """
    调整图像到目标尺寸，保持内容比例，使用黑色填充，并保存到指定路径。

    参数:
        image (PIL.Image): 输入图像。
        save_path (str): 保存图像的路径。
        target_size (tuple): 目标尺寸，默认是 (512, 512)。

    返回:
        str: 保存的图像路径。
    """
    original_width, original_height = image.size
    target_width, target_height = target_size

    # 判断是否需要缩放
    if original_width > target_width or original_height > target_height:
        # 找到需要缩放到 512 的比例
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 填充图像到 512x512
    delta_w = target_width - image.size[0]
    delta_h = target_height - image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    new_image = ImageOps.expand(image, padding, fill=0)

    # 保存处理后的图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保保存目录存在
    new_image.save(save_path)

    return save_path
import os
import numpy as np
from PIL import Image


def blend_images_using_mask(image1_path, image2_path, mask_path, save_path):
    """
    使用二值掩码将两张图像进行裁剪并合并。

    参数:
        image1_path (str): 第一张图像路径（前景图）。
        image2_path (str): 第二张图像路径（背景图）。
        mask_path (str): 掩码图像路径（原始应该为0和255的二值图）。
        save_path (str): 合成图像的保存路径。

    返回:
        str: 保存的合成图像路径。
    """
    # 读取图像
    image1 = Image.open(image1_path).convert("RGBA")
    image2 = Image.open(image2_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")

    # 确保尺寸一致
    image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
    mask = mask.resize(image1.size, Image.Resampling.LANCZOS)

    # 优化掩码：先用中值滤波去除噪点，再阈值二值化
    mask = mask.filter(ImageFilter.MedianFilter(size=3))
    mask = mask.point(lambda p: 255 if p > 128 else 0)

    # 转换为 numpy 数组
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    mask_array = np.array(mask)

    # 生成透明前景：保留 mask == 255 的区域
    foreground = np.zeros_like(image1_array, dtype=np.uint8)
    foreground[mask_array == 255] = image1_array[mask_array == 255]

    # 生成透明背景：保留 mask == 0 的区域
    background = np.zeros_like(image2_array, dtype=np.uint8)
    background[mask_array == 0] = image2_array[mask_array == 0]

    # 合并前景和背景
    blended_image = Image.alpha_composite(Image.fromarray(background), Image.fromarray(foreground))

    # 保存合成结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    blended_image.save(save_path)

    return save_path

def handls(pic_name, pic_type):
    # /Users/dean/Desktop/xxxxddd.jpeg
    img_path = "/Users/dean/Downloads/" + pic_name + pic_type
    image = Image.open(img_path).convert("RGB")
    out_pic = 'output_masks/' + pic_name + pic_type
    img_path = resize_with_padding(image, out_pic)
    image = Image.open(img_path).convert("RGB")
    mask_image_path = save_binary_clothing_mask(img_path)
    mask_image = Image.open(mask_image_path).convert("L")
    pippres = CustomInpaintPipeline()
    img = pippres.generate_image(image=image, mask_image=mask_image, control_image=None,
                                 controlnet_conditioning_scale=None)

    save_name = pic_name +'-org'+'.png'
    img.save(save_name)
    return save_name,out_pic, mask_image_path

if __name__ == '__main__':
    # IMG_4924.PNG IMG_4909.JPG IMG_4938.PNG IMG_4924.PNG
    pic_name = "IMG_4924"
    pic_type = ".PNG"
    save_name, out_pic, mask_image_path = handls(pic_name, pic_type)
    blend_images_using_mask(save_name, out_pic, mask_image_path, "./" + pic_name + ".png")



