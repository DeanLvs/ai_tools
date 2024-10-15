import random
import numpy as np
from PIL import Image
import torch
import cv2
from controlnet_aux import MidasDetector, ZoeDetector
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import io
import numpy as np
import pickle
import asyncio,sys,os
import zipfile
from typing import Optional
import threading

app = FastAPI()

class ZoeProcessorSingleton:
    _instance = None
    _lock = threading.Lock()  # Ensure thread-safe singleton access

    @classmethod
    def get_instance(cls, device="cuda"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check locking to avoid race conditions
                    print(f'------加载 ZoeDetector 到----- {device}')
                    cls._instance = ZoeDetector.from_pretrained("lllyasviel/Annotators")
                    cls._instance = cls._instance.to(device)
                    print(f'------加载完成 ZoeDetector 到----- {device}')
        return cls._instance

def get_controlnet_img(img_3d, img_org):
    # 随机选择使用 Zoe 或 Midas 处理图像
    # if random.random() > 0.5:
    #     controlnet_img = self.processor_zoe(img, output_type='cv2')
    # else:
        # controlnet_img = self.processor_midas(img, output_type='cv2')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor_zoe = ZoeProcessorSingleton.get_instance(device)
    def gen_it(tem_img):
        controlnet_img = processor_zoe(tem_img, output_type='cv2')
        # 调整图像大小
        height, width, _ = controlnet_img.shape
        ratio = np.sqrt(1024. * 1024. / (width * height))
        new_width, new_height = int(width * ratio), int(height * ratio)
        controlnet_img = cv2.resize(controlnet_img, (new_width, new_height))
        controlnet_img = Image.fromarray(controlnet_img)
        return controlnet_img
    if img_3d is not None:
        g_img_3d = gen_it(img_3d)
    else:
        g_img_3d = None
    g_img_org = gen_it(img_org)
    return g_img_3d, g_img_org


async def restart_program():
    """Restarts the current program."""
    print("Restarting program...")
    # await asyncio.sleep(50)  # 延迟5秒，确保响应已发送
    python_executable = sys.executable  # 使用当前 Python 的可执行文件路径
    os.execv(python_executable, [python_executable] + sys.argv)
@app.get("/re")
async def inpaint(
    tasks: BackgroundTasks
):
    print('re start depth es ----------------')
    # 重启示范内存
    tasks.add_task(restart_program)

@app.post("/inpaint")
async def inpaint(
    file_i: Optional[UploadFile] = File(None),
    file_next: Optional[UploadFile] = File(None)
):
    print('Start processing...')
    try:
        # 读取上传的文件并转换为 PIL.Image
        if file_i is not None:
            image_pil_3d = Image.open(file_i.file).convert("RGB")
        else:
            image_pil_3d = None
        next_image_pil_org = Image.open(file_next.file).convert("RGB")

        normal_map_img_3d, next_normal_map_img = get_controlnet_img(image_pil_3d, next_image_pil_org)
        if normal_map_img_3d is not None:
            # 调整图像大小与原始输入一致
            normal_map_img_3d = normal_map_img_3d.resize(image_pil_3d.size)
            # 将结果通过 pickle 序列化
            masks_io_3d = io.BytesIO()
            normal_map_img_3d.save(masks_io_3d, format="PNG")
            masks_io_3d.seek(0)
        else:
            masks_io_3d = None
        next_normal_map_img = next_normal_map_img.resize(next_image_pil_org.size)

        # 将结果通过 pickle 序列化
        next_masks_io = io.BytesIO()
        next_normal_map_img.save(next_masks_io, format="PNG")
        next_masks_io.seek(0)

        # 创建 ZIP 文件并将 d_pose 和 masks 打包
        zip_io = io.BytesIO()
        with zipfile.ZipFile(zip_io, mode="w") as zf:
            if masks_io_3d is not None:
                # 添加 d_pose_resized 到 ZIP
                zf.writestr("depth_3d.png", masks_io_3d.getvalue())
            # 添加 masks 到 ZIP
            zf.writestr("depth_org.png", next_masks_io.getvalue())

        zip_io.seek(0)
        torch.cuda.empty_cache()
        # 返回 ZIP 文件作为响应
        re_it = StreamingResponse(zip_io, media_type="application/zip",
                                  headers={"Content-Disposition": "attachment; filename=result.zip"})
        # 返回 ZIP 文件
        return re_it

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5070)
