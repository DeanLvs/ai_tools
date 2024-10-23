import argparse
from PIL import Image
import zipfile
from typing import List, Optional
import time, gc, io
from pydantic import BaseModel
import numpy as np
import cv2
import os, torch, traceback, gc, io
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
app = FastAPI()
from src.flux.xflux_pipeline import XFluxPipeline

class FluxArguments:
    def __init__(self):
        self.prompt = ""
        self.neg_prompt = ""
        self.img_prompt = None
        self.neg_img_prompt = None
        self.ip_scale = 1.0
        self.neg_ip_scale = 1.0
        self.local_path = None
        self.repo_id = None
        self.name = None
        self.ip_repo_id = 'XLabs-AI/flux-ip-adapter'
        self.ip_name = 'flux-ip-adapter.safetensors'
        self.ip_local_path = None
        self.lora_repo_id = None
        self.lora_name = None
        self.lora_local_path = None
        self.device = "cuda"
        self.offload = False
        self.use_ip = True
        self.use_lora = False
        self.use_controlnet = False
        self.num_images_per_prompt = 1
        self.image = None
        self.lora_weight = 0.9
        self.control_weight = 0.8
        self.control_type = None
        self.model_type = "flux-dev"
        self.width = 1024
        self.height = 1024
        self.num_steps = 25
        self.guidance = 4
        self.seed = 123456789
        self.true_gs = 4.5
        self.timestep_to_start_cfg = 5
        self.save_path = "results"

def gen_img(args: FluxArguments):
    if args.image:
        image = Image.open(args.image)
    else:
        image = None
    # sapianfNudeMenWomen_v20FP16.safetensors 和lora flux_realism_lora.safetensors      还有 fuxCapacityNSFWPornFlux_v03Bf16Unet.safetensors cust_pat='STOIQOA/STOIQOAfroditeFLUXXL_F1DAlpha.safetensors' , cust_lb=['tf']
    # xflux_pipeline = XFluxPipeline(args.model_type, args.device, args.offload, cust_pat='STOIQOA/STOIQOAfroditeFLUXXL_F1DAlpha.safetensors')
    xflux_pipeline = XFluxPipeline(args.model_type, args.device, args.offload)
    if args.use_ip:
        print('load ip-adapter:', args.ip_local_path, args.ip_repo_id, args.ip_name)
        xflux_pipeline.set_ip(args.ip_local_path, args.ip_repo_id, args.ip_name)
    if args.use_lora:
        print('load lora:', args.lora_local_path, args.lora_repo_id, args.lora_name)
        xflux_pipeline.set_lora(args.lora_local_path, args.lora_repo_id, args.lora_name, args.lora_weight)
    if args.use_controlnet:
        print('load controlnet:', args.local_path, args.repo_id, args.name)
        xflux_pipeline.set_controlnet(args.control_type, args.local_path, args.repo_id, args.name)

    image_prompt = Image.open(args.img_prompt) if args.img_prompt else None
    neg_image_prompt = Image.open(args.neg_img_prompt) if args.neg_img_prompt else None
    images = []
    for _ in range(args.num_images_per_prompt):
        result = xflux_pipeline(
            prompt=args.prompt,
            controlnet_image=image,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed,
            true_gs=args.true_gs,
            control_weight=args.control_weight,
            neg_prompt=args.neg_prompt,
            timestep_to_start_cfg=args.timestep_to_start_cfg,
            image_prompt=image_prompt,
            neg_image_prompt=neg_image_prompt,
            ip_scale=args.ip_scale,
            neg_ip_scale=args.neg_ip_scale,
        )
        images.append(result)
        args.seed = args.seed + 1
    return images


class RequestDataMul(BaseModel):
    # data = {
    #     'file_path': file_path,
    #     'prompt': prompt,
    #     'only_file_path': only_face_path,
    #     'seed': seed,
    #     'gen_type': gen_type
    # }
    file_path: str
    prompt: str
    only_file_path: str
    seed: int
    gen_type: str
# 保存OpenCV图像为BytesIO
def save_cv2_image_to_bytesio(final_img):
    img_io = io.BytesIO()
    success, encoded_image = cv2.imencode('.png', final_img)
    if success:
        img_io.write(encoded_image.tobytes())
        img_io.seek(0)
        return img_io
    else:
        raise ValueError("Failed to encode image using OpenCV.")
async def restart_program():
    """Restarts the current program by executing the shell script."""
    print("Restarting program...")
    # 指定你的 .sh 脚本的路径
    shell_script_path = "/nvme0n1-disk/flux/x-flux/start.sh"  # 修改为你的 .sh 脚本的路径
    # 使用 os.execv 来执行 .sh 脚本，替换当前进程
    os.execv("/bin/bash", ["bash", shell_script_path])

@app.get("/re")
async def inpaint_report(
    tasks: BackgroundTasks
):
    print('re start depose es ----------------')
    # 重启示范内存
    tasks.add_task(restart_program)

@app.post('/inpaint')
async def process_pics(
        file_path: str = Form(...),  # 从表单接收文件路径
        prompt: str = Form(...),  # 从表单接收 prompt 字符串
        seed: int = Form(...),
        gen_type: Optional[str] = Form(None),
        only_file_path: Optional[str] = Form(None)
):
    try:
        # if only_file_path is not None and only_file_path != '':
        #     file_path = only_file_path
        arguments = FluxArguments()
        arguments.prompt = prompt
        arguments.img_prompt = file_path
        arguments.seed=seed
        total_pic = gen_img(arguments)

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
        print('suc gen img ------')
        return re_it

    except Exception as e:
        # 打印完整的错误堆栈信息
        print("Error occurred:")
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1006)
