import os
import torch.backends.cudnn as cudnn
import torch
import cv2
from fastapi.responses import StreamingResponse
import asyncio, io, sys, pickle
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Depends
app = FastAPI()
from skimage.transform import warp
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
import numpy as np
import threading

class PixieSingleton:
    _instance = None
    _visualizer = None
    _lock = threading.Lock()

    @classmethod
    def get_pixie_instance(cls, device='cuda:0'):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Initialize PIXIE and Visualizer with thread safety
                    print(f"------加载 PIXIE 到 {device}-----")
                    pixie_cfg.model.use_tex = False
                    cls._instance = PIXIE(config=pixie_cfg, device=device)
                    print(f"------加载完成 PIXIE 到 {device}-----")
        return cls._instance

    @classmethod
    def get_visualizer_instance(cls, device='cuda:0'):
        if cls._visualizer is None:
            with cls._lock:
                if cls._visualizer is None:
                    print(f"------加载 Visualizer 到 {device}-----")
                    pixie_cfg.model.use_tex = False
                    cls._visualizer = Visualizer(render_size=1024, config=pixie_cfg, device=device, rasterizer_type='pytorch3d')
                    print(f"------加载完成 Visualizer 到 {device}-----")
        return cls._visualizer

def run_it(inputpath):
    # 直接设置输入图片路径和保存文件夹路径
    device = 'cuda:0'
    # 检查环境
    if not torch.cuda.is_available():
        print('CUDA 不可用！使用 CPU')
        device = 'cpu'
    else:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
    # 加载测试图片
    testdata = TestData(inputpath, iscrop=True, body_detector='rcnn')
    # 运行 PIXIE
    pixie_cfg.model.use_tex = False
    # Retrieve PIXIE and Visualizer instances from the singleton
    pixie = PixieSingleton.get_pixie_instance(device=device)
    visualizer = PixieSingleton.get_visualizer_instance(device=device)
    # 拟合给定图片的 SMPLX 模型
    batch = testdata[0]
    util.move_dict_to_device(batch, device)
    batch['image'] = batch['image'].unsqueeze(0)
    batch['image_hd'] = batch['image_hd'].unsqueeze(0)
    batch['original_image'] = batch['original_image'].unsqueeze(0)
    tform_hd = batch['tform_hd']
    shp = batch['shp']
    data = {'body': batch}
    # 进行编码和解码
    param_dict = pixie.encode(data)
    input_codedict = param_dict['body']
    input_opdict = pixie.decode(input_codedict, param_type='body')
    input_opdict['albedo'] = visualizer.tex_flame2smplx(input_opdict['albedo'])
    visdict_dis = visualizer.render_results(input_opdict, data['body']['image_hd'], overlay=False)

    def vis_pic(temp_visdict):
        # 确保 shape_images 的数据格式和范围正确
        grid_image_all = temp_visdict['shape_images'].squeeze(0)  # 移除批处理维度
        grid_image_all_t = grid_image_all.permute(1, 2, 0).cpu().numpy()  # 转换为 NumPy 数组 (H, W, C)
        grid_image_all = (grid_image_all_t * 255).astype(np.uint8)  # 归一化到 0-255 范围
        return grid_image_all, grid_image_all_t
    only_grid_image_all, only_grid_image_all_t = vis_pic(visdict_dis)

    # 还原图像到原始尺寸
    grid_image_all_org = warp(only_grid_image_all_t, tform_hd, output_shape=shp)
    grid_image_all_bgr = (grid_image_all_org * 255).astype(np.uint8)

    # 获取图像的尺寸
    height, width, _ = grid_image_all_bgr.shape
    box_left, box_right, box_top, box_bottom = batch['box_index']
    # 遍历框外的区域，将纯黑色替换为纯白色
    # 顶部和底部区域
    for y in range(height):
        for x in range(width):
            # 判断是否在框外
            if y <= box_top or y >= box_bottom or x <= box_left or x >= box_right:
                # 替换为纯白色
                grid_image_all_bgr[y, x] = [255, 255, 255]
    white_mask = np.all(grid_image_all_bgr == [255, 255, 255], axis=-1)
    grid_image_all_bgr[white_mask] = [0, 0, 0]
    return grid_image_all_bgr


async def restart_program():
    """Restarts the current program by executing the shell script."""
    print("Restarting program...")
    # 指定你的 .sh 脚本的路径
    shell_script_path = "/nvme0n1-disk/transBody/PIXIE/run.sh"  # 修改为你的 .sh 脚本的路径
    # 使用 os.execv 来执行 .sh 脚本，替换当前进程
    os.execv("/bin/bash", ["bash", shell_script_path])

@app.get("/re")
async def inpaint(
    tasks: BackgroundTasks
):
    print('re start depose es ----------------')
    # 重启示范内存
    tasks.add_task(restart_program)

@app.post("/inpaint")
async def inpaint(file_path: str = Form(...)):
    # dense 分析
    try:
        # 使用 file_path 读取并处理图像
        result_image = run_it(file_path)

        # 提取文件扩展名，判断是 jpg 还是 png
        _, file_extension = os.path.splitext(file_path.lower())
        file_extension = file_extension.replace('.', '')  # 移除前缀的 '.'

        # 支持的图像格式
        valid_extensions = {'jpg', 'jpeg', 'png'}
        if file_extension not in valid_extensions:
            return {"error": "Unsupported file format. Only jpg and png are allowed."}

        # 将结果转换为字节流
        masks_io = io.BytesIO()
        if file_extension in ['jpg', 'jpeg']:
            _, buffer = cv2.imencode('.jpg', result_image)
        elif file_extension == 'png':
            _, buffer = cv2.imencode('.png', result_image)
        masks_io.write(buffer)
        masks_io.seek(0)
        torch.cuda.empty_cache()
        t = StreamingResponse(masks_io, media_type=f"image/{file_extension}")
        # 返回图像，设置相应的 MIME 类型
        return t
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)