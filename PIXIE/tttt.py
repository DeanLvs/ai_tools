import os
import torch.backends.cudnn as cudnn
import torch
import cv2
from skimage.transform import estimate_transform, warp, resize, rescale
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
import numpy as np
from PIL import ImageDraw,Image, ImageFilter
from torchvision import transforms
from PIL import Image

from skimage.transform import AffineTransform
import numpy as np
import torch
from skimage.transform import warp


def transform_hd_to_original(image_hd, tform_hd, original_size):
    """
    将 image_hd 转换回原始图像大小，使用逆向变换矩阵。

    参数:
        image_hd (torch.Tensor): 某个分辨率的图像张量 (C, H, W)，由原始图像通过变换生成。
        tform_hd (torch.Tensor): 用于将原始图像变换为 image_hd 的变换矩阵。
        original_size (tuple): 原始图像的大小 (height, width)。

    返回:
        torch.Tensor: 转换回与原始图像大小匹配的图像。
    """
    # 将 tform_hd 从 torch.Tensor 转换为 NumPy 数组
    tform_hd_np = tform_hd.numpy()

    # 确保 tform_hd 是有效的仿射变换矩阵
    transform_matrix = AffineTransform(matrix=tform_hd_np)

    # 使用逆向变换矩阵将图像还原
    hd_image_np = image_hd.permute(1, 2, 0).numpy()  # 将 torch.Tensor 转换为 NumPy 数组 (H, W, C)

    # 使用 inverse map 恢复原始图像
    restored_image_np = warp(hd_image_np, inverse_map=transform_matrix.inverse, output_shape=original_size)

    # 将还原后的图像转换回 torch.Tensor (C, H, W)
    restored_image = torch.tensor(restored_image_np.transpose(2, 0, 1)).float()

    return restored_image

def save_restored_image(restored_image, save_path):
    """
    将张量格式的图像保存为文件

    参数:
        restored_image (torch.Tensor): 需要保存的恢复后的图像 (C, H, W)
        save_path (str): 保存图像的路径
    """
    # 将图像从 (C, H, W) 转换为 (H, W, C)
    restored_image_np = restored_image.permute(1, 2, 0).numpy()

    # 将 numpy 数组转换为 PIL 图像
    restored_image_pil = Image.fromarray((restored_image_np * 255).astype(np.uint8))  # 将图像值从 [0,1] 乘以 255 转换为 [0,255]

    # 保存图像
    restored_image_pil.save(save_path)
    print(f'Image saved to {save_path}')
def resize_image(image_path, max_size=2048, file_name=""): #
    image = Image.open(image_path+file_name)

    width, height = image.size
    if width > max_size or height > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        if image.mode == 'P':
            image = image.convert('RGB')
        image.save(f'/Users/dean/PycharmProjects/Body/PIXIE/{file_name}')
        width, height = image.size
    return height, width

def main():

    # 直接设置输入图片路径和保存文件夹路径
    inputpath_org = '/Users/dean/Desktop/uploads/'
    file_name = 'w700d1q75cms.jpg'
    h,w = resize_image(image_path=inputpath_org, max_size=1000, file_name=file_name)
    inputpath = '/Users/dean/PycharmProjects/Body/PIXIE/'+file_name
    savefolder = 'TestSamples/output'
    device = 'cuda:0'

    os.makedirs(savefolder, exist_ok=True)
    device = 'cpu'

    # 加载测试图片
    testdata = TestData(inputpath, iscrop=True, body_detector='rcnn', device='cpu')

    # 拟合给定图片的 SMPLX 模型
    batch = testdata[0]
    util.move_dict_to_device(batch, device)

    image_hd = batch['image_hd']
    tform_hd = batch['tform_hd']
    original_image = batch['original_image']

    # 获取原始图像大小
    original_size = original_image.shape[1:]  # (height, width)

    # 还原回原始图像
    restored_image = transform_hd_to_original(image_hd, tform_hd, original_size)
    save_restored_image(restored_image,'xxxxxxx.jpg')

if __name__ == '__main__':
    main()