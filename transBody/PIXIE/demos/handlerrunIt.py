import os
import torch.backends.cudnn as cudnn
import torch
import cv2

from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
import numpy as np
from PIL import Image

def main():
    # 直接设置输入图片路径和保存文件夹路径
    inputpath = 'IMG_5047.png'
    savefolder = 'TestSamples/output'
    device = 'cuda:0'

    # 打开图片
    image = Image.open(inputpath)

    # 获取图片的宽度和高度
    width, height = image.size

    print(f"Image Width: {width}, Image Height: {height}")
    os.makedirs(savefolder, exist_ok=True)

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
    pixie = PIXIE(config=pixie_cfg, device=device)
    visualizer = Visualizer(h_size=height,w_size=width, config=pixie_cfg, device=device, rasterizer_type='pytorch3d')

    # 拟合给定图片的 SMPLX 模型
    batch = testdata[0]
    util.move_dict_to_device(batch, device)
    batch['image'] = batch['image'].unsqueeze(0)
    batch['image_hd'] = batch['image_hd'].unsqueeze(0)
    name = batch['name']
    data = {'body': batch}

    param_dict = pixie.encode(data)
    input_codedict = param_dict['body']

    # 可视化并渲染结果
    input_opdict = pixie.decode(input_codedict, param_type='body')
    # 通过过滤顶点，去除手和头部
    # input_opdict:(['vertices', 'transformed_vertices', 'face_kpt', 'smplx_kpt', 'smplx_kpt3d', 'joints', 'cam', 'albedo'])

    input_opdict['albedo'] = visualizer.tex_flame2smplx(input_opdict['albedo'])
    visdict = visualizer.render_results(input_opdict, data['body']['image_hd'], overlay=True)
    input_image = batch['image_hd'].clone()
    input_shape = visdict['shape_images'].clone()


    # 确保 shape_images 的数据格式和范围正确
    grid_image_all = visdict['shape_images'].squeeze(0)  # 移除批处理维度
    grid_image_all = grid_image_all.permute(1, 2, 0).cpu().numpy()  # 转换为 NumPy 数组
    grid_image_all = (grid_image_all * 255).astype(np.uint8)  # 归一化到 0-255 范围
    # cv2.imwrite(os.path.join(savefolder, f'{name}_processed.jpg'), grid_image_all)

    # 使用 OpenCV 保存图像
    cv2.imwrite(f'{name}_processed.jpg', grid_image_all)
    print(f'图像已保存到 {name}')

if __name__ == '__main__':
    main()