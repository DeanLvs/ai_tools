import cv2
import numpy as np
import base64, traceback
import torch,os
import gc
from flask import Flask, request, jsonify
import io
from SimSwapIt import SimSwap
from SimSwapMultispecific import swap_multi
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from VideoSwapMultispecific import swap_multi_video
import json
import zipfile
app = FastAPI()

# 创建 Pydantic 模型来接收请求数据
class ImageDataModel(BaseModel):
    image_data: str
    image_data_n: str

@app.post("/process_image")
async def process_image(request: Request, image_data_model: ImageDataModel):
    try:

        # 处理图像
        image_data = image_data_model.image_data
        image_data_n = image_data_model.image_data_n

        # 解码图像数据
        img_binary = base64.b64decode(image_data)
        nparr = np.frombuffer(img_binary, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 解码图像数据
        img_binary_n = base64.b64decode(image_data_n)
        nparr_n = np.frombuffer(img_binary_n, np.uint8)
        img_n = cv2.imdecode(nparr_n, cv2.IMREAD_COLOR)

        simswap = SimSwap()

        # Single face swap example
        final_img = simswap.swap_faces(img_b_whole =img_n, img_a_whole = img)
        # 将处理后的 NumPy 数组编码为 JPEG 格式
        _, img_encoded = cv2.imencode('.jpg', final_img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    finally:
        # 释放 GPU 内存
        del simswap
        torch.cuda.empty_cache()
        gc.collect()
    # 返回处理后的图像数据
    return JSONResponse(content={'processed_image': img_base64})
from typing import List, Optional
class RequestDataMul(BaseModel):
    pic_b: str
    pic_save: Optional[str] = None
    source_path_list: List[str]
    target_path_list: List[str]
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
    shell_script_path = "/nvme0n1-disk/swapface/SimSwap/start.sh"  # 修改为你的 .sh 脚本的路径
    # 使用 os.execv 来执行 .sh 脚本，替换当前进程
    os.execv("/bin/bash", ["bash", shell_script_path])

@app.get("/re")
async def inpaint_report(
    tasks: BackgroundTasks
):
    print('re start depose es ----------------')
    # 重启示范内存
    tasks.add_task(restart_program)

@app.post('/process_image_mul')
async def process_pics(data: RequestDataMul):
    try:
        # 假设 `swap_multi` 是一个返回 OpenCV 图像的函数
        final_img = swap_multi(data.pic_b, data.source_path_list, data.target_path_list)

        # 创建一个 BytesIO 对象，作为 ZIP 文件的内存存储
        zip_io = io.BytesIO()

        # 创建 ZIP 文件
        with zipfile.ZipFile(zip_io, mode='w') as zip_file:
            img_io = save_cv2_image_to_bytesio(final_img)
            # 使用 pic_save 参数，若没有提供则使用默认文件名
            file_name = 'final_img.png'
            # 将图片添加到 ZIP 文件中
            zip_file.writestr(file_name, img_io.getvalue())

        # 设置 ZIP 文件的起始位置
        zip_io.seek(0)
        re_it = StreamingResponse(zip_io, media_type="application/zip",
                                  headers={"Content-Disposition": "attachment; filename=images.zip"})
        print('成功生成图像 ------')
        return re_it
    except Exception as e:
        # 打印完整的错误堆栈信息
        print("Error occurred:")
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        gc.collect()

@app.post('/process_video_mul')
async def process_video(data: RequestDataMul):
    try:
        # 假设 `swap_multi` 是一个返回 OpenCV 图像的函数
        final_video_path = swap_multi_video(data.pic_b, data.pic_save, data.source_path_list, data.target_path_list)
        print('成功生成图像 ------')
        # 返回 JSON 响应，其中包含视频的路径
        return {"status": "success", "video_path": final_video_path}
    except Exception as e:
        # 打印完整的错误堆栈信息
        print("Error occurred:")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)