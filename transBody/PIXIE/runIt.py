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
from skimage.io import imread
from skimage.transform import warp
from PIL import Image


def run_it(inputpath):
    # 直接设置输入图片路径和保存文件夹路径
    device = 'cpu'  # 使用 CPU

    # 加载测试图片
    testdata = TestData(inputpath, iscrop=True, body_detector='rcnn', device='cpu')

    # 运行 PIXIE
    pixie_cfg.model.use_tex = False
    pixie_cfg.device = 'cpu'
    pixie_cfg.device_id = None
    pixie = PIXIE(config=pixie_cfg, device='cpu')
    visualizer = Visualizer(render_size=1024, config=pixie_cfg, device=device, rasterizer_type='pytorch3d')
    batch = testdata[0]
    util.move_dict_to_device(batch, device)
    batch['image'] = batch['image'].unsqueeze(0)
    batch['image_hd'] = batch['image_hd'].unsqueeze(0)
    batch['original_image'] = batch['original_image'].unsqueeze(0)
    name = batch['name']
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
                grid_image_all_bgr[y, x] = [255, 255, 255]
        # 将图像中的纯白色变成纯黑色
    white_mask = np.all(grid_image_all_bgr == [255, 255, 255], axis=-1)
    grid_image_all_bgr[white_mask] = [0, 0, 0]

    # 保存还原后的图像
    restored_image_path = f'/Users/dean/PycharmProjects/Body/PIXIE/{name}_oo.jpg'
    cv2.imwrite(restored_image_path, grid_image_all_bgr)


if __name__ == '__main__':
    run_it("/Users/dean/PycharmProjects/Body/PIXIE/IMG_5248.png")

