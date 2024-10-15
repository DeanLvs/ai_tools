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

def calculate_black_edge(image, axis=0):
    """
    计算图像沿给定轴的纯黑色边缘的长度
    - axis=0: 垂直方向(列，左右边缘)
    - axis=1: 水平方向(行，上下边缘)
    """
    if axis == 0:  # 计算左右边缘的黑色区域长度
        black_lengths = []
        for col in range(image.shape[1]):
            if np.all(image[:, col] == 0):  # 检查这一列是否全黑
                black_lengths.append(col)
            else:
                break
        return len(black_lengths)
    elif axis == 1:  # 计算上下边缘的黑色区域长度
        black_lengths = []
        for row in range(image.shape[0]):
            if np.all(image[row, :] == 0):  # 检查这一行是否全黑
                black_lengths.append(row)
            else:
                break
        return len(black_lengths)


def get_crop_coordinates(grid_image_all):
    """
    计算黑色边缘的裁剪坐标，返回上下和左右的裁剪起始和结束位置。
    """
    # 转换为灰度图来检测黑色区域
    grayscale_image = cv2.cvtColor(grid_image_all, cv2.COLOR_RGB2GRAY)

    # 计算左侧和右侧的黑色边缘长度
    left_black_length = calculate_black_edge(grayscale_image, axis=0)
    right_black_length = calculate_black_edge(np.fliplr(grayscale_image), axis=0)  # 镜像计算右边
    max_side_crop = max(left_black_length, right_black_length)

    # 计算顶部和底部的黑色边缘长度
    top_black_length = calculate_black_edge(grayscale_image, axis=1)
    bottom_black_length = calculate_black_edge(np.flipud(grayscale_image), axis=1)  # 镜像计算底部
    max_top_bottom_crop = max(top_black_length, bottom_black_length)

    # 返回裁剪区域的坐标
    return max_side_crop, max_top_bottom_crop

def apply_crop(image, max_side_crop, max_top_bottom_crop):
    """
    根据给定的裁剪区域裁剪图像。
    """
    return image[max_top_bottom_crop:image.shape[0] - max_top_bottom_crop,
                 max_side_crop:image.shape[1] - max_side_crop]


def add_individual_padding_inplace(image, add_left, add_right, add_top, add_bottom):
    """
    直接在原图的上下左右分别增加白色像素。
    - add_left: 增加左侧的像素比例（百分比）。
    - add_right: 增加右侧的像素比例（百分比）。
    - add_top: 增加顶部的像素比例（百分比）。
    - add_bottom: 增加底部的像素比例（百分比）。
    """
    # 计算原始图像的尺寸
    original_height, original_width = image.shape[:2]

    # 计算每个方向要添加的像素数
    left_padding = int(original_width * add_left)
    right_padding = int(original_width * add_right)
    top_padding = int(original_height * add_top)
    bottom_padding = int(original_height * add_bottom)

    # 使用OpenCV的copyMakeBorder来增加边框，边框颜色为白色 (255, 255, 255)
    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding,
                                      cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return padded_image

def resize_and_pad(image, target_size=(1024, 1024)):
    """
    将图片调整为指定尺寸，保持原始宽高比，如果超过目标尺寸则缩小，并用黑色填充不足区域。
    - image: 输入图像，类型为 NumPy 数组。
    - target_size: 目标尺寸，默认 (1024, 1024)。
    """
    target_width, target_height = target_size

    # 获取原始图像的尺寸
    original_height, original_width = image.shape[:2]

    # 计算调整尺寸的比例
    scale = min(target_width / original_width, target_height / original_height)

    # 计算调整后的尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 调整图像尺寸
    resized_image = cv2.resize(image, (new_width, new_height))

    # 创建黑色背景图像
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 计算图像在背景中的位置
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # 将调整后的图像放入黑色背景图像中
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return padded_image


def crop_and_save(image_path, box_left, box_right, box_top, box_bottom, save_path='xxxxxx123.png'):
    """
    从图像中截取指定区域并保存。
    - image_path: 输入的图像路径。
    - box_left: 裁剪区域的左边界。
    - box_right: 裁剪区域的右边界。
    - box_top: 裁剪区域的上边界。
    - box_bottom: 裁剪区域的下边界。
    - save_path: 保存裁剪后图像的路径。
    """
    # 加载图像
    image = cv2.imread(image_path)

    # 确保边界在图像范围内
    box_left = int(max(0, box_left))
    box_right = int(min(image.shape[1], box_right))
    box_top = int(max(0, box_top))
    box_bottom = int(min(image.shape[0], box_bottom))

    # 截取图像区域
    cropped_image = image[box_top:box_bottom, box_left:box_right]

    # 保存裁剪后的图像
    cv2.imwrite(save_path, cropped_image)
    print(f'裁剪后的图像已保存到 {save_path}')
    
    return cropped_image

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
    # 拟合给定图片的 SMPLX 模型
    batch = testdata[0]
    util.move_dict_to_device(batch, device)
    batch['image'] = batch['image'].unsqueeze(0)
    (add_left, add_right, add_top, add_bottom, box_left, box_right, box_top, box_bottom) = batch['add_s']
    cropped_image = crop_and_save(inputpath, box_left, box_right, box_top, box_bottom)
    batch['image_hd'] = batch['image_hd'].unsqueeze(0)
    batch['original_image'] = batch['original_image'].unsqueeze(0)
    name = batch['name']
    data = {'body': batch}

    # 进行编码和解码
    param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
    # pixie.decode_Tpose(param_dict)
    input_codedict = param_dict['body']
    # 获取形状参数
    shape_params = input_codedict['shape']
    # 更新形状参数
    input_codedict['shape'] = shape_params

    input_opdict = pixie.decode(input_codedict, param_type='body')

    input_opdict['albedo'] = visualizer.tex_flame2smplx(input_opdict['albedo'])

    # 将图像调整为 1024x1024
    result_image = resize_and_pad(cropped_image, (1024, 1024))

    # 调整图像维度
    result_image = result_image.transpose(2, 0, 1)  # 转换为 (channels, height, width)

    # 转换为 PyTorch 张量，并指定数据类型
    result_image_tensor = torch.tensor(result_image, dtype=torch.float32)

    # 增加 batch 维度
    result_image_tensor = result_image_tensor.unsqueeze(0)

    visdict = visualizer.render_results(input_opdict, result_image_tensor, overlay=True)

    visdict_dis = visualizer.render_results(input_opdict, result_image_tensor, overlay=False)

    def vis_pic(temp_visdict):
        # 确保 shape_images 的数据格式和范围正确
        grid_image_all = temp_visdict['shape_images'].squeeze(0)  # 移除批处理维度
        grid_image_all = grid_image_all.permute(1, 2, 0).cpu().numpy()  # 转换为 NumPy 数组 (H, W, C)
        grid_image_all = (grid_image_all * 255).astype(np.uint8)  # 归一化到 0-255 范围
        return grid_image_all

    grid_image_all = vis_pic(visdict)
    only_grid_image_all = vis_pic(visdict_dis)

    # 计算裁剪区域
    max_side_crop, max_top_bottom_crop = get_crop_coordinates(grid_image_all)

    # 裁剪图像
    grid_image_all = apply_crop(grid_image_all, max_side_crop, max_top_bottom_crop)
    only_grid_image_all = apply_crop(only_grid_image_all, max_side_crop, max_top_bottom_crop)
    org_image_all = add_individual_padding_inplace(grid_image_all, add_left, add_right, add_top, add_bottom)
    only_org_image_all = add_individual_padding_inplace(only_grid_image_all, add_left, add_right, add_top, add_bottom)
    # 保存裁剪后的图像
    cv2.imwrite(f'/Users/dean/PycharmProjects/Body/PIXIE/{name}_processed.jpg', org_image_all)
    print(f'未裁剪图像已保存到 /Users/dean/PycharmProjects/Body/PIXIE/{name}_processed.jpg')


    # 保存 only_grid_image_all 裁剪后的图像
    cv2.imwrite(f'/Users/dean/PycharmProjects/Body/PIXIE/{name}_only_processed.jpg', only_org_image_all)
    print(f'裁剪后的 only_grid_image_all 已保存到 /Users/dean/PycharmProjects/Body/PIXIE/{name}_only_processed.jpg')


if __name__ == '__main__':
    run_it("/Users/dean/Downloads/IMG_5248.png")

