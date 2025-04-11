import cv2
import numpy as np
import base64, traceback
import torch,os
import gc
from flask import Flask, request, jsonify
import io
from face2face import Face2Face

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json
import zipfile
app = FastAPI()

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
    shell_script_path = "/nvme0n1-disk/ai_tools/SocAIty/face2face/start.sh"  # 修改为你的 .sh 脚本的路径
    # 使用 os.execv 来执行 .sh 脚本，替换当前进程
    os.execv("/bin/bash", ["bash", shell_script_path])

@app.get("/re")
async def inpaint_report(
    tasks: BackgroundTasks
):
    print('re start depose es ----------------')
    # 重启示范内存
    tasks.add_task(restart_program)

def get_swap_map(f2f, data):
    swap_map = {}
    for idx, source_face_path in enumerate(data.source_path_list):
        face_name = f's_{str(idx)}'
        embedding = f2f.add_face(face_name, source_face_path, save=True)
        swap_map[face_name] = f't_{str(idx)}'
    for idx, target_face_path in enumerate(data.target_path_list):
        embedding = f2f.add_face(f't_{str(idx)}', target_face_path, save=True)
    return swap_map


@app.post('/process_image_mul')
async def process_pics(data: RequestDataMul):
    try:
        f2f = Face2Face(device_id=0)
        swap_map = get_swap_map(f2f, data)
        final_img = f2f.swap(
            media=data.pic_b,
            faces=swap_map,
            enhance_face_model='gfpgan_1.4'
        )
        cv2.imwrite('/nvme0n1-disk/book_yes/temps.png', final_img)
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

@app.post('/process_video_mul')
async def process_video(data: RequestDataMul):
    try:
        # 假设 `swap_multi` 是一个返回 OpenCV 图像的函数
        f2f = Face2Face(device_id=0)
        swap_map = get_swap_map(f2f, data)
        final_video = f2f.swap(
            media=data.pic_b,
            faces=swap_map,
            enhance_face_model='gpen_bfr_512'
        )
        final_video.save(data.pic_save)
        print('成功生成图像 ------')
        # 返回 JSON 响应，其中包含视频的路径
        return {"status": "success", "video_path": data.pic_save}
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
    uvicorn.run(app, host="0.0.0.0", port=5007)