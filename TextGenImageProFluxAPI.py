import torch, cv2, requests
import matplotlib.pyplot as plt
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from transformers import CLIPTextModel, T5EncoderModel
from safetensors.torch import load_file
from diffusers.pipelines import StableDiffusionXLPipeline, AutoPipelineForText2Image, FluxControlNetPipeline, FluxPipeline, FluxControlNetInpaintPipeline, FluxInpaintPipeline, FluxControlNetImg2ImgPipeline
from PIL import Image
from diffusers.models import FluxMultiControlNetModel
from controlnet_aux import NormalBaeDetector
from diffusers.utils import load_image
from book_yes_logger_config import logger

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
def progress_callback(pipeline_instance, step, t, latents, had, room_id, num_inference_steps):
    # 仅在每 5 个步骤调用一次
    if step == 0 or step % 5 == 0 or step == num_inference_steps - 1:  # 或在最后一步调用
        progress = int((step / num_inference_steps) * 100)
        call_main_service(progress, had, room_id)
    return latents
def create_callback(had_value, room_id, num_inference_steps):
    # lambda 函数接受 4 个参数，以配合 FluxPipeline 的 callback_on_step_end 调用
    return lambda pipeline_instance, step, t, latents: progress_callback(pipeline_instance, step, t, latents, had_value, room_id, num_inference_steps)

def transNormal(in_image_path):
    normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")

    normal_bae.to("cuda")
    im = Image.open(in_image_path)
    surface = normal_bae(im)

    # Convert surface normal to a PIL image if it's a tensor
    if isinstance(surface, torch.Tensor):
        surface = surface.squeeze(0).cpu().numpy().astype("uint8")
        control_image = Image.fromarray(surface)
    else:
        control_image = surface
    return control_image

def canny_face(face_path):
    # 'mf_1_8d38164b-4d4a-4688-b9f5-9883f9e5c88a.png'
    # 读取已经提取出的人脸图像
    face_image = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
    # 对图像进行高斯模糊，减少噪声
    # blurred_face = cv2.GaussianBlur(face_image, (3, 3), 0)

    # 应用 Canny 边缘检测，调节阈值
    edges = cv2.Canny(face_image, 50, 150)  # 使用较低的阈值，生成较细的边缘

    # 将 NumPy 数组转换为三通道图像
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # 将 NumPy 数组转换为 PIL Image 对象
    edges_pil = Image.fromarray(edges_rgb)
    return edges_pil

def gen_img_normals_control(control_image, prompt):
    # Load pipeline
    controlnet = FluxControlNetModel.from_pretrained(
      "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
      torch_dtype=torch.bfloat16
    )
    safetensors_file = '/nvme0n1-disk/civitai-downloader/STOIQOA/STOIQOAfroditeFLUXXL_F1DAlpha.safetensors'
    pipe = FluxControlNetPipeline.from_single_file(
      pretrained_model_link_or_path=safetensors_file,
      controlnet=controlnet,
      torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    image = pipe(
        prompt,
        control_image=control_image,
        controlnet_conditioning_scale=0.2,
        num_inference_steps=28,
        guidance_scale=3.5,
        height=1024,
        width=1024
    ).images[0]
    return image

def gen_img_canny_control_with_face(control_image_path, prompt):
    control_image = canny_face(control_image_path)
    return gen_img_canny_control(control_image, prompt)
def gen_img_and_swap_face(prompt, room_id=None):
    return text_get_img(prompt, room_id)

def gen_img_canny_control(control_image, prompt):
    # Load pipeline
    controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Canny'
    controlnet = FluxControlNetModel.from_pretrained(
      controlnet_model,
      torch_dtype=torch.bfloat16
    )
    safetensors_file = '/nvme0n1-disk/civitai-downloader/STOIQOA/STOIQOAfroditeFLUXXL_F1DAlpha.safetensors'
    pipe = FluxControlNetPipeline.from_single_file(
      pretrained_model_link_or_path=safetensors_file,
      controlnet=controlnet,
      torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    image = pipe(
        prompt,
        control_image=control_image,
        controlnet_conditioning_scale=0.1,
        num_inference_steps=30,
        guidance_scale=3.5,
        height=1024,
        width=1024
    )[0]
    return image
import os
def text_get_img(prompt, room_id=None):
    os.environ["HF_TOKEN"] = "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"
    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    safetensors_file = '/nvme0n1-disk/civitai-downloader/STOIQOA/STOIQOAfroditeFLUXXL_F1DAlpha.safetensors'
    # 加载 FLUX 专用的 CLIP L 文本编码器
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder_2 = T5EncoderModel.from_pretrained('google/t5-v1_1-xxl', from_tf=True)
    pipe = FluxPipeline.from_single_file(safetensors_file, torch_dtype=torch.bfloat16, text_encoder=text_encoder, text_encoder_2=text_encoder_2)

    # state_dict = load_file(safetensors_file)

    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        callback_on_step_end=create_callback('1/2', room_id, 50),
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image

def ip_text_get_img(prompt, ip_image):
    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    safetensors_file = '/nvme0n1-disk/civitai-downloader/STOIQOA/STOIQOAfroditeFLUXXL_F1DAlpha.safetensors'
    pipe = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.load_ip_adapter('XLabs-AI/flux-ip-adapter')
    # state_dict = load_file(safetensors_file)
    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    image = pipe(
        prompt,
        ip_adapter_image=ip_image,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image

def gen_img_inpaint_control(image, mask_image, prompt, control_image_pose):
    # including canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6).
    controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
    controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
    controlnet = FluxMultiControlNetModel(
        [controlnet_union])  # we always recommend loading via FluxMultiControlNetModel
    controlnet.config = controlnet_union.config

    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    safetensors_file = '/nvme0n1-disk/civitai-downloader/STOIQOA/STOIQOAfroditeFLUXXL_F1DAlpha.safetensors'

    pipe = FluxControlNetInpaintPipeline.from_single_file(safetensors_file, controlnet=controlnet, torch_dtype=torch.bfloat16)
    # state_dict = load_file(safetensors_file)

    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    image = pipe(
        prompt,
        control_image=[control_image_pose], #control_image_depth
        control_mode=[4], #2,0,
        image=image,
        mask_image=mask_image,
        guidance_scale=7,
        strength=0.7,
        num_inference_steps=40,
        max_sequence_length=512,
        controlnet_conditioning_scale=[0.5], # 0.1, ,0.3
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image

def inpaint_get_img_control(image, mask_image, prompt, control_image_canny, control_image_depth, control_image_pose):
    # including canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6).
    controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
    controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
    controlnet = FluxMultiControlNetModel(
        [controlnet_union])  # we always recommend loading via FluxMultiControlNetModel
    controlnet.config = controlnet_union.config

    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    safetensors_file = '/nvme0n1-disk/civitai-downloader/STOIQOA/STOIQOAfroditeFLUXXL_F1DAlpha.safetensors'

    pipe = FluxControlNetInpaintPipeline.from_single_file(safetensors_file, controlnet=controlnet, torch_dtype=torch.bfloat16)
    # state_dict = load_file(safetensors_file)

    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    image = pipe(
        prompt,
        control_image=[control_image_pose], #control_image_depth
        control_mode=[4], #2,0,
        image=image,
        mask_image=mask_image,
        guidance_scale=7,
        strength=0.7,
        num_inference_steps=40,
        max_sequence_length=512,
        controlnet_conditioning_scale=[0.5], # 0.1, ,0.3
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image

def text_get_img_control(image, mask_image, prompt, control_image_canny, control_image_depth, control_image_pose):
    # including canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6).
    controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
    controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
    controlnet = FluxMultiControlNetModel(
        [controlnet_union])  # we always recommend loading via FluxMultiControlNetModel
    controlnet.config = controlnet_union.config

    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    safetensors_file = '/nvme0n1-disk/civitai-downloader/STOIQOA/STOIQOAfroditeFLUXXL_F1DAlpha.safetensors'

    pipe = FluxControlNetInpaintPipeline.from_single_file(safetensors_file, controlnet=controlnet, torch_dtype=torch.bfloat16)
    # state_dict = load_file(safetensors_file)

    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    image = pipe(
        prompt,
        control_image=[control_image_pose], #control_image_depth
        control_mode=[0], #2,0,
        image=image,
        mask_image=mask_image,
        guidance_scale=7,
        strength=0.7,
        num_inference_steps=40,
        max_sequence_length=512,
        controlnet_conditioning_scale=[0.5], # 0.1, ,0.3
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image


def inpaint_get_img(image, mask_image, prompt):
    # including canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6).

    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    safetensors_file = '/nvme0n1-disk/civitai-downloader/STOIQOA/STOIQOAfroditeFLUXXL_F1DAlpha.safetensors'

    pipe = FluxInpaintPipeline.from_single_file(safetensors_file, torch_dtype=torch.bfloat16)
    # state_dict = load_file(safetensors_file)
    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    image = pipe(
        prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=3.5,
        strength=0.966,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image


def test_it():
    # control_image = transNormal('/nvme0n1-disk/book_yes/static/uploads/7104360038/mf_2_dc6035f3-1db0-439c-9139-873a5ad5ebb1.png')
    # control_image = transNormal('/nvme0n1-disk/book_yes/static/uploads/7104360038/mf_1_8d38164b-4d4a-4688-b9f5-9883f9e5c88a.png')
    # control_image.save('xxx9999dd.png')
    # image = gen_img_normals_control(control_image, "a sexy woman, naked, wearing blue stockings and high heels, standing in the crowd, facing the camera")
    # image.save('normals_x_normals_xxx.png')
    # image = text_get_img("a sexy woman, naked, wearing blue stockings and high heels, standing in the crowd")
    # image.save('sexxxxxxxx.png')
    # input_image = Image.open('/nvme0n1-disk/book_yes/static/uploads/7104360038/filled_image_pil_f9c279c5-e99e-4b9d-9b7c-4157d7d8cb7c.jpg')
    # input_image = Image.open('/nvme0n1-disk/book_yes/static/uploads/7104360038/clear_return_f9c279c5-e99e-4b9d-9b7c-4157d7d8cb7c.jpg')
    # input_image = Image.open('/nvme0n1-disk/book_yes/static/uploads/7104360038/f9c279c5-e99e-4b9d-9b7c-4157d7d8cb7c.jpg')
    # mask_image  = Image.open('/nvme0n1-disk/book_yes/static/uploads/7104360038/with_torso_mask_f9c279c5-e99e-4b9d-9b7c-4157d7d8cb7c.jpg')
    # mask_image  = Image.open('/nvme0n1-disk/book_yes/static/uploads/7104360038/with_torso_mask_f9c279c5-e99e-4b9d-9b7c-4157d7d8cb7c.jpg')
    # prompt = 'sexy nude woman, authenticity, had natural asses and natural boobs'
    # control_image_canny = Image.open('/nvme0n1-disk/book_yes/static/uploads/7104360038/line_auto_f9c279c5-e99e-4b9d-9b7c-4157d7d8cb7c.jpg')
    # control_image_depth = Image.open('/nvme0n1-disk/book_yes/static/uploads/7104360038/nor_control_net_f9c279c5-e99e-4b9d-9b7c-4157d7d8cb7c.jpg')
    # control_image_pose = Image.open('/nvme0n1-disk/book_yes/static/uploads/7104360038/control_net_f9c279c5-e99e-4b9d-9b7c-4157d7d8cb7c.jpg')
    #
    # image = inpaint_get_img_control(input_image, mask_image, prompt, control_image_canny, control_image_depth, control_image_pose)
    # image = image.resize((input_image.size), Image.LANCZOS)
    # image.save('inpaintxxx.png')
    # image = inpaint_get_img(image, mask_image, prompt)
    # image.save('inpaintxxxno.png')

    # image = ip_text_get_img('sexy woman dance', Image.open('/nvme0n1-disk/book_yes/static/uploads/7104360038/f9c279c5-e99e-4b9d-9b7c-4157d7d8cb7c.jpg'))
    # image.save('ipipipipip.png')
    # control_image = canny_face('/nvme0n1-disk/book_yes/static/uploads/7104360038/mf_1_8d38164b-4d4a-4688-b9f5-9883f9e5c88a.png')
    # control_image.save('canny_xxxcanny_canny.png')
    # image = gen_img_canny_control(control_image, "a sexy woman, naked, wearing blue stockings and high heels, standing in the crowd, facing the camera")
    # image.save('canny_canny_canny.png')
    return

