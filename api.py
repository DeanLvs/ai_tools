from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
import torch
import sys
from PIL import Image
import asyncio
from io import BytesIO
import os
import random
from glob import glob
import gc

from pipeline import StableVideoDiffusionPipeline
from safetensors import safe_open
from animatelcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler
from diffusers.utils import load_image, export_to_video

app = FastAPI()

# 初始化调度器
noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
    num_train_timesteps=40,
    sigma_min=0.002,
    sigma_max=700.0,
    sigma_data=1.0,
    s_noise=1.0,
    rho=7,
    clip_denoised=False,
)

# 模型选择函数
def model_select(pipe, selected_file):
    print("Loading model weights:", selected_file)
    pipe.unet.cpu()
    file_path = os.path.join("./safetensors", selected_file)
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=True)
    del state_dict
    return

# 加载和推理函数
def load_model_to_gpu():
    # 加载模型到 GPU
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        scheduler=noise_scheduler,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    model_select(pipe, "AnimateLCM-SVD-xt-1.1.safetensors")
    pipe.to("cuda")
    return pipe

# 卸载模型函数
# def unload_model_from_gpu(pipe):
#     del pipe
#     torch.cuda.empty_cache()
#     gc.collect()

async def restart_program():
    """Restarts the current program."""
    print("Restarting program...")
    await asyncio.sleep(1)  # 延迟5秒，确保响应已发送
    python_executable = sys.executable  # 使用当前 Python 的可执行文件路径
    os.execv(python_executable, [python_executable] + sys.argv)
# 视频生成函数
def sample(
    pipe,
    image: Image,
    seed: int = 42,
    randomize_seed: bool = False,
    motion_bucket_id: int = 80,
    fps_id: int = 8,
    max_guidance_scale: float = 1.2,
    min_guidance_scale: float = 1,
    width: int = 1024,
    height: int = 576,
    num_inference_steps: int = 4,
    decoding_t: int = 4,
    output_folder: str = "outputs_api",
):
    if image.mode == "RGBA":
        image = image.convert("RGB")

    if randomize_seed:
        seed = random.randint(0, 2**63 - 1)
    generator = torch.manual_seed(seed)

    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

    with torch.no_grad():
        frames = pipe(
            image,
            decode_chunk_size=decoding_t,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
        ).frames[0]

    # 将生成的帧导出为视频
    export_to_video(frames, video_path, fps=fps_id)
    torch.manual_seed(seed)

    return video_path, seed

@app.post("/generate-video/")
async def generate_video(
    file: UploadFile = File(...),
    seed: int = Form(42),
    randomize_seed: bool = Form(False),
    motion_bucket_id: int = Form(80),
    fps_id: int = Form(8),
    max_guidance_scale: float = Form(1.2),
    min_guidance_scale: float = Form(1),
    width: int = Form(1024),
    height: int = Form(576),
    num_inference_steps: int = Form(4),
    decoding_t: int = Form(4)
):
    image = Image.open(BytesIO(await file.read()))

    # 加载模型到 GPU
    pipe = load_model_to_gpu()

    # 执行推理
    video_path, seed = sample(
        pipe=pipe,
        image=image,
        seed=seed,
        randomize_seed=randomize_seed,
        motion_bucket_id=motion_bucket_id,
        fps_id=fps_id,
        max_guidance_scale=max_guidance_scale,
        min_guidance_scale=min_guidance_scale,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        decoding_t=decoding_t,
    )

    # 卸载模型并释放 GPU 资源
    # unload_model_from_gpu(pipe)
    # 创建一个后台任务来延迟重启
    return_it = FileResponse(video_path, media_type="video/mp4")
    asyncio.create_task(restart_program())
    return return_it

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)