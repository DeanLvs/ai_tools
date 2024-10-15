import torch
from diffusers import FluxControlNetInpaintPipeline, FluxInpaintPipeline
from diffusers.models import FluxControlNetModel
from PIL import Image
from book_yes_logger_config import logger
from CoreService import GenContextParam
class FluxInpaintPipelineSingleton:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info(f'start init FluxInpaintPipeline')
            cls._instance = super(FluxInpaintPipelineSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell", controlnet_list_t = ['canny'], use_out_test = False, torch_dtype=torch.float16, device="cuda"):
        if not self._initialized:
            try:
                # 初始化模型
                logger.info(f'start init FluxInpaintPipeline {model_name}')
                self.controlnet_list = controlnet_list_t
                self.model_name = model_name
                self.use_out_test = use_out_test
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipe = self.init_pipe()
                self._initialized = True
            except Exception as e:
                # 捕获所有异常并打印详细的错误信息
                logger.info(f"Error during initialization: {e}")
                raise
    def init_pipe(self):
        if len(self.controlnet_list) == 0:
            logger.info(f'init ----------------------------- use Normal Pipeline ')
            pipe = FluxInpaintPipeline.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, use_safetensors=False)
            pipe.to(self.device)
            return pipe
        controlnet_init_list = []
        for controlnet_item in self.controlnet_list:
            if 'key_points' == controlnet_item:
                # 加载 OpenPose 的 ControlNet 模型
                # controlnet_openpose = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16).to(self.device)
                # controlnet_init_list.append(controlnet_openpose)
                controlnet_openpose = FluxControlNetModel.from_pretrained("Shakker-Labs/FLUX.1-dev-ControlNet-Pose",torch_dtype=torch.float16)
                controlnet_init_list.append(controlnet_openpose)
                logger.info('none pose')
            elif 'depth' == controlnet_item:
                controlnet_dept = FluxControlNetModel.from_pretrained("Shakker-Labs/FLUX.1-dev-ControlNet-Depth", torch_dtype=torch.bfloat16)
                controlnet_init_list.append(controlnet_dept)
            elif 'seg' == controlnet_item:
                controlnet_seg = None
                controlnet_init_list.append(controlnet_seg)
            elif 'canny' == controlnet_item:
                controlnet_canny = FluxControlNetModel.from_pretrained(
                    "InstantX/FLUX.1-dev-controlnet-canny", torch_dtype=torch.float16
                )
                controlnet_init_list.append(controlnet_canny)
        # 创建自定义管道
        if len(controlnet_init_list) == 0:
            to_controlnet = None
        elif len(controlnet_init_list) == 1:
            to_controlnet = controlnet_init_list[0]
        else:
            to_controlnet = controlnet_init_list
        logger.info(f'init ----------------------------- use ControlNet Pipeline{len(controlnet_init_list)}')
        pipe = FluxControlNetInpaintPipeline.from_pretrained(
            self.model_name, controlnet=to_controlnet, torch_dtype=torch.float16, use_safetensors=False
        )
        pipe.to(self.device)
        return pipe

    def genIt(self, g_c_param: GenContextParam):
        if len(self.controlnet_list) == 0:
            logger.info(
                f'run ----------------------------- use Inpaint Pipeline {g_c_param}')
            resultTemp = self.pipe(
                prompt=g_c_param.prompt,
                prompt_2=g_c_param.prompt_2,
                image=g_c_param.next_genImage,
                mask_image=g_c_param.big_mask,
                strength=g_c_param.strength,
                num_inference_steps=g_c_param.num_inference_steps,
                callback_on_step_end=g_c_param.func_call_back,
                guidance_scale=g_c_param.guidance_scale,
            ).images[0]
        else:
            control_image_list_t = []
            controlnet_conditioning_scale_t = []
            logger.info(f'use {self.controlnet_list}')
            for cont in self.controlnet_list:
                if 'key_points' == cont:
                    # control_image_list_t.append(g_c_param.control_img_list[0])
                    # controlnet_conditioning_scale_t.append(g_c_param.control_float_array[0])
                    logger.info('none pose')
                elif 'depth' == cont:
                    if g_c_param.control_img_list[1]:
                        logger.info(f'use depth line pic---------')
                        control_image_list_t.append(g_c_param.control_img_list[1])
                        controlnet_conditioning_scale_t.append(g_c_param.control_float_array[1])
                    else:
                        logger.info(f'use def none depth pic pic---------')
                        t_img_width, t_img_height = g_c_param.next_genImage.size
                        control_image_list_t.append(Image.new('RGB', (t_img_width, t_img_height), 0))
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
                        control_image_list_t.append(Image.new('RGB', (t_img_width, t_img_height), 0))
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
                prompt = g_c_param.prompt,
                prompt_2 = g_c_param.prompt_2,
                image=g_c_param.next_genImage,
                mask_image=g_c_param.big_mask,
                control_image=control_image_list_f,
                controlnet_conditioning_scale=controlnet_conditioning_scale_f,
                callback_on_step_end=g_c_param.func_call_back,
                strength=g_c_param.strength,
                num_inference_steps=g_c_param.num_inference_steps,
                guidance_scale=g_c_param.guidance_scale,
            ).images[0]
        return resultTemp, ''
