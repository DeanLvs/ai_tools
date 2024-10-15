from controlnet_aux import OpenposeDetector
from PIL import Image
import torch

class SingletonOpenposeDetector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonOpenposeDetector, cls).__new__(cls)
            # 初始化 OpenPose 检测器
            cls._instance.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            cls._instance.openpose.to(torch.device("cpu"))
        return cls._instance

    def detect(self, image):
        # 使用 OpenPose 检测骨架图像
        return self.openpose(image)

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import io
import numpy as np
import pickle
import asyncio,sys,os

app = FastAPI()
async def restart_program():
    """Restarts the current program."""
    print("Restarting program...")
    # await asyncio.sleep(3)  # 延迟5秒，确保响应已发送
    python_executable = sys.executable  # 使用当前 Python 的可执行文件路径
    os.execv(python_executable, [python_executable] + sys.argv)

@app.get("/re")
async def inpaint(
    tasks: BackgroundTasks
):
    print('re start pose es ----------------')
    # 重启示范内存
    tasks.add_task(restart_program)

@app.post("/inpaint")
async def inpaint(
    file_i: UploadFile = File(...)  # 接收 mask_clear 图像
):
    print('start open_pose es ----------------')
    try:
        # 将上传文件转换为 PIL.Image
        image_pil = Image.open(file_i.file).convert("RGB")
        # 使用 get_normal_pic 函数生成法线图
        singleton_openpose = SingletonOpenposeDetector()
        # 检测并调整 control_image 大小
        control_image = singleton_openpose.detect(image_pil)
        control_image = control_image.resize(image_pil.size)
        # 将掩码转换为字节流 (pickle 序列化)
        masks_io = io.BytesIO()
        pickle.dump(control_image, masks_io)
        masks_io.seek(0)
        torch.cuda.empty_cache()
        return_it = StreamingResponse(masks_io, media_type="application/octet-stream")
        print('finish open_pose es ----------------')
        # 返回字节流响应
        return return_it
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5060)


