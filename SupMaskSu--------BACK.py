#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import logging
import os
import sys
from typing import Any, ClassVar, Dict, List
import torch
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.extractor import create_extractor

import traceback
from sklearn.cluster import KMeans
class DensePoseModel:
    _instance = None

    def __new__(cls, config_file, model_file, device="cpu"):
        if cls._instance is None:
            cls._instance = super(DensePoseModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file, model_file, device="cpu"):
        if self._initialized:
            return
        self.cfg = self.setup_config(config_file, model_file, device)
        self.predictor = DefaultPredictor(self.cfg)
        self.logger = setup_logger(name="DensePoseModel")
        self._initialized = True

    def setup_config(self, config_file, model_file, device):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_file
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        return cfg

    def generate_mask(self, img, parts_to_include=None):
        img_height, img_width = img.shape[:2]
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]

        # 使用 fine segmentation visualizer 生成掩码
        visualizer = DensePoseResultsFineSegmentationVisualizer(self.cfg)
        extractor = create_extractor(visualizer)
        data = extractor(outputs)
        # 创建一个全黑的掩码
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if parts_to_include is not None:
            for instance_idx, instance_data in enumerate(data[0]):  # 遍历所有实例
                labels = instance_data.labels.cpu().numpy()
                bbox = outputs.pred_boxes.tensor[instance_idx].cpu().numpy().astype(int)
                x0, y0, x1, y1 = bbox
                part_mask = np.zeros_like(labels, dtype=np.uint8)
                for part in parts_to_include:
                    part_mask = np.maximum(part_mask, (labels == part).astype(np.uint8) * 255)

                # 调整掩码大小以匹配原始图像
                if (x1 - x0) > 0 and (y1 - y0) > 0:
                    part_mask_resized = cv2.resize(part_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
                    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], part_mask_resized)
                else:
                    self.logger.warning(f"Invalid bounding box dimensions: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
        return mask

    def generate_m_mask(self, img, parts_to_include=None):
        img_height, img_width = img.shape[:2]
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]

        # 使用 fine segmentation visualizer 生成掩码
        visualizer = DensePoseResultsFineSegmentationVisualizer(self.cfg)
        extractor = create_extractor(visualizer)
        data = extractor(outputs)

        # 创建字典来存储不同部分的掩码
        masks = {key: np.zeros((img_height, img_width), dtype=np.uint8) for key in parts_to_include}

        if parts_to_include is not None:
            for instance_idx, instance_data in enumerate(data[0]):  # 遍历所有实例
                labels = instance_data.labels.cpu().numpy()
                bbox = outputs.pred_boxes.tensor[instance_idx].cpu().numpy().astype(int)
                x0, y0, x1, y1 = bbox

                for key, parts in parts_to_include.items():
                    part_mask = np.zeros_like(labels, dtype=np.uint8)
                    for part in parts:
                        part_mask = np.maximum(part_mask, (labels == part).astype(np.uint8) * 255)

                    # 调整掩码大小以匹配原始图像
                    if (x1 - x0) > 0 and (y1 - y0) > 0:
                        part_mask_resized = cv2.resize(part_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
                        masks[key][y0:y1, x0:x1] = np.maximum(masks[key][y0:y1, x0:x1], part_mask_resized)
                    else:
                        self.logger.warning(f"Invalid bounding box dimensions: x0={x0}, y0={y0}, x1={x1}, y1={y1}")

        return masks

def convert_to_ndarray(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError("Unsupported image type")

def handlerMask(img_path):
    img = read_image(img_path, format="BGR")
    # 配置文件和模型文件路径
    config_file = "detectron2/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml"
    model_file = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2_s1x/173294801/model_final_6e1ed1.pkl"

    # 判断是否可以使用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化 DensePose 模型
    densepose_model = DensePoseModel(config_file, model_file, device)
    # 移除 ,23, 24
    ###fine
    # segmentation: 1, 2 = Torso, 3 = Right
    # Hand, 4 = Left
    # Hand,
    # 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
    # 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right,
    # 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
    # 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left,
    # 20, 22 = Lower Arm Right, 23, 24 = Head
    parts_to_include = [3,4,5,6]
    mask = densepose_model.generate_mask(img, parts_to_include=parts_to_include)
    return mask
def subtract_masks(mask1, mask2):
    mask1 = convert_to_ndarray(mask1)
    mask2 = convert_to_ndarray(mask2)

    # 确保两个掩码的尺寸相同
    if mask1.shape != mask2.shape:
        raise ValueError("The shapes of mask1 and mask2 do not match.")

    # 计算重叠区域（即 mask1 和 mask2 都为非零的部分）
    overlap = (mask1 > 0) & (mask2 > 0)

    # 对于 mask2 覆盖的部分，mask1 减去 mask2
    result_mask = np.where(overlap, mask1 - mask2, mask1)

    # 确保结果中没有负值
    subtracted_mask = np.maximum(result_mask, 0)

    return subtracted_mask

def apply_skin_tone_gap(image, masks, mask_A, skin_tones = {
    "body": (213, 239, 255),  # 较浅的肤色
    "leg": (181, 228, 181),   # 中等肤色
    "arm": (196, 228, 255)    # 中等浅肤色
}, non_fill_percentage=1):
    if image is None:
        raise ValueError("Image cannot be None")

    if mask_A is None:
        raise ValueError("Mask A cannot be None")

    # 将结果图像初始化为输入图像
    result_image_like = image.copy()

    # 确保 mask_A 为布尔数组
    mask_A = mask_A.astype(bool)

    # 定义一个全为 False 的掩码，用来保存所有的交界处
    boundary_mask = np.zeros(image.shape[:2], dtype=bool)

    # 计算总像素数的2%
    total_pixels = image.shape[0] * image.shape[1]
    non_fill_pixels = int(total_pixels * (non_fill_percentage / 100))

    # 遍历每个部分，应用对应的肤色填充
    for part, mask in masks.items():
        if part in skin_tones:
            skin_color = skin_tones[part]
            mask_bool = mask.astype(bool)

            # 计算当前部分和所有已处理部分的交界处
            current_boundary = np.logical_and(boundary_mask, mask_bool)
            boundary_mask = np.logical_or(boundary_mask, mask_bool)

            # 确定交界处扩展的次数，直到达到所需的不填充像素数量
            dilate_iterations = 0
            while np.sum(current_boundary) < non_fill_pixels and dilate_iterations < 100:
                dilate_iterations += 1
                kernel = np.ones((3, 3), np.uint8)
                current_boundary = cv2.dilate(current_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)

            # 只处理同时存在于 mask 和 mask_A 中的像素，且不在扩展的交界处
            combined_mask = np.logical_and(mask_bool, mask_A)
            combined_mask = np.logical_and(combined_mask, ~current_boundary)

            # 将非交界处的区域填充为指定的肤色
            result_image_like[combined_mask] = skin_color

    return result_image_like

def is_skin_pixel_hsv(pixel, lower_bound, upper_bound):
    return all(lower_bound <= pixel) and all(pixel <= upper_bound)

def extract_skin_pixels_hsv(image_np, mask_bool):
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 58, 89], dtype=np.uint8)
    upper_skin = np.array([30, 174, 255], dtype=np.uint8)
    masked_image = hsv_image[mask_bool]
    skin_pixels = [pixel for pixel in masked_image if is_skin_pixel_hsv(pixel, lower_skin, upper_skin)]
    return np.array(skin_pixels)

def adjust_color_brightness(color, factor):
    """
    调整颜色的亮度。
    :param color: BGR 颜色元组 (B, G, R)
    :param factor: 亮度因子 (<1.0 表示变暗, >1.0 表示变亮)
    :return: 调整后的 BGR 颜色
    """
    adjusted_color = np.clip(np.array(color) * factor, 0, 255).astype(np.uint8)
    return tuple(adjusted_color)

def generate_body_part_colors(average_color, num_parts=5):
    """
    生成不同身体部位的颜色。
    :param average_color: 平均皮肤颜色 (B, G, R)
    :param num_parts: 部位数量
    :return: 部位颜色列表
    """
    factors = np.linspace(0.9, 1.1, num_parts)  # 从稍微变暗到稍微变亮
    body_part_colors = [adjust_color_brightness(average_color, factor) for factor in factors]
    return body_part_colors

def calculate_average_color(image_pil, mask1, mask2):
    image_np = np.array(image_pil)
    subtracted_mask = subtract_masks(mask1, mask2)
    mask_bool = subtracted_mask.astype(bool)
    skin_pixels = extract_skin_pixels_hsv(image_np, mask_bool)
    average_skin_color_hsv = np.mean(skin_pixels, axis=0) if len(skin_pixels) > 0 else (0, 0, 0)

    # 将平均颜色从 HSV 转换为 BGR
    if len(skin_pixels) > 0:
        average_skin_color_hsv = np.uint8([[average_skin_color_hsv]])  # 转换为合适的形状和类型
        average_skin_color_bgr = cv2.cvtColor(average_skin_color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(average_skin_color_bgr)
    else:
        return (196,228,255)


def apply_skin_tone_co(org_img, image, masks, mask_A, skin_tones = {
    "Torso_1": 1,
    "Torso_2": 1,
    # "Right_Hand_3": (196, 228, 255),
    # "Left_Hand_4": (196, 228, 255),
    # "Left_Foot_5": (181, 228, 181),
    # "Right_Foot_6": (181, 228, 181),
    "Upper_Leg_Right_7": 2,
    "Upper_Leg_Left_8": 3,
    "Upper_Leg_Right_9": 2,
    "Upper_Leg_Left_10": 3,
    "Lower_Leg_Right_11": 2,
    "Lower_Leg_Left_12": 3,
    "Lower_Leg_Right_13": 2,
    "Lower_Leg_Left_14": 3,
    "Upper_Arm_Left_15": 4,
    "Upper_Arm_Right_16": 5,
    "Upper_Arm_Left_17": 4,
    "Upper_Arm_Right_18": 5,
    "Lower_Arm_Left_19": 4,
    "Lower_Arm_Right_20": 5,
    "Lower_Arm_Left_21": 4,
    "Lower_Arm_Right_22": 5,
    # "Head_23": (195, 145, 130),
    # "Head_24": (195, 145, 130)
}):
    if image is None:
        raise ValueError("Image cannot be None")

    if mask_A is None:
        raise ValueError("Mask A cannot be None")
    body_part_colors = (196,228,255)
    try:
        img_width, img_height = org_img.size
        # 创建一个全零的数组用于存储最终的合并掩码
        combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # 迭代所有掩码，将它们叠加到 combined_mask 上
        for mask in masks.values():
            combined_mask = np.maximum(combined_mask, mask)
        skin_mask = subtract_masks(combined_mask, mask_A)
        average_skin_color_bgr = calculate_average_color(org_img, skin_mask, combined_mask)
        print(f"average_skin_color_bgr :{average_skin_color_bgr}")

    except Exception as ex:
        print(ex)
        # 打印完整的异常堆栈
        traceback.print_exc()
    body_part_colors = generate_body_part_colors(average_skin_color_bgr)
    result_image_like = image
        # 确保mask_A为布尔数组
    mask_A = mask_A.astype(bool)
    print("hunm filter")
    # 遍历每个部分，应用对应的肤色填充
    for part, mask in masks.items():
        if part in skin_tones:
            skin_color = skin_tones[part]
            mask_bool = mask.astype(bool)

            # 只处理同时存在于mask和mask_A中的像素
            combined_mask = np.logical_and(mask_bool, mask_A)
            result_image_like[combined_mask] = body_part_colors[skin_color - 1]

    return result_image_like

def apply_skin_tone(org_img, image, masks, mask_A, skin_tones = {
    "Torso_1": 1,
    "Torso_2": 1,
    # "Right_Hand_3": (196, 228, 255),
    # "Left_Hand_4": (196, 228, 255),
    # "Left_Foot_5": (181, 228, 181),
    # "Right_Foot_6": (181, 228, 181),
    "Upper_Leg_Right_7": 2,
    "Upper_Leg_Left_8": 3,
    "Upper_Leg_Right_9": 2,
    "Upper_Leg_Left_10": 3,
    "Lower_Leg_Right_11": 2,
    "Lower_Leg_Left_12": 3,
    "Lower_Leg_Right_13": 2,
    "Lower_Leg_Left_14": 3,
    "Upper_Arm_Left_15": 4,
    "Upper_Arm_Right_16": 5,
    "Upper_Arm_Left_17": 4,
    "Upper_Arm_Right_18": 5,
    "Lower_Arm_Left_19": 4,
    "Lower_Arm_Right_20": 5,
    "Lower_Arm_Left_21": 4,
    "Lower_Arm_Right_22": 5,
    # "Head_23": (195, 145, 130),
    # "Head_24": (195, 145, 130)
}):
    if image is None:
        raise ValueError("Image cannot be None")

    if mask_A is None:
        raise ValueError("Mask A cannot be None")
    average_skin_color_bgr = (196,228,255)
    body_part_colors = generate_body_part_colors(average_skin_color_bgr)
    result_image_like = image
        # 确保mask_A为布尔数组
    mask_A = mask_A.astype(bool)
    print("hunm filter")
    # 遍历每个部分，应用对应的肤色填充

    for part, mask in masks.items():
        if part in skin_tones:
            skin_color = skin_tones[part]
            mask_bool = mask.astype(bool)

            # 只处理同时存在于mask和mask_A中的像素
            combined_mask = np.logical_and(mask_bool, mask_A)
            result_image_like[combined_mask] = body_part_colors[skin_color - 1]

    return result_image_like

def handlerMaskAll(img_path):
    img = read_image(img_path, format="BGR")
    # 配置文件和模型文件路径
    config_file = "detectron2/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml"
    model_file = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2_s1x/173294801/model_final_6e1ed1.pkl"

    # 判断是否可以使用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化 DensePose 模型
    densepose_model = DensePoseModel(config_file, model_file, device)
    # 移除 ,23, 24
    ###fine
    # segmentation: 1, 2 = Torso, 3 = Right
    # Hand, 4 = Left
    # Hand,
    # 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
    # 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right,
    # 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
    # 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left,
    # 20, 22 = Lower Arm Right, 23, 24 = Head
    # parts_to_include = [3,4,5,6]
    body_to_include = [1, 2]
    leg_to_include  = [7,9, 8,10, 11,13, 12,14]
    arm_to_include = [15,17, 16,18, 19,21, 20,22]
    head_to_include = [23,24]
    hand_to_include = [3, 4]
    foot_to_include = [5, 6]

    parts_to_include = {
        "Torso_1": [1],
        "Torso_2": [2],
        "Upper_Leg_Right_7": [7],
        "Upper_Leg_Left_8": [8],
        "Upper_Leg_Right_9": [9],
        "Upper_Leg_Left_10": [10],
        "Lower_Leg_Right_11": [11],
        "Lower_Leg_Left_12": [12],
        "Lower_Leg_Right_13": [13],
        "Lower_Leg_Left_14": [14],
        "Upper_Arm_Left_15": [15],
        "Upper_Arm_Right_16": [16],
        "Upper_Arm_Left_17": [17],
        "Upper_Arm_Right_18": [18],
        "Lower_Arm_Left_19": [19],
        "Lower_Arm_Right_20": [20],
        "Lower_Arm_Left_21": [21],
        "Lower_Arm_Right_22": [22],
        "exclude": [3,4,5,6,23,24,],
        # "all_body": [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    }
    mask = densepose_model.generate_m_mask(img, parts_to_include=parts_to_include)
    return mask


def merge_masks(part_masks):
    merged_masks = defaultdict(lambda: np.zeros_like(next(iter(part_masks.values()))))

    for part_name, mask in part_masks.items():
        # 提取部分名称的前缀（例如 "Torso_", "Upper_Arm_Left_"）
        prefix = '_'.join(part_name.split('_')[:-1])  # 获取前缀部分

        # 合并具有相同前缀的掩码
        merged_masks[prefix] = np.maximum(merged_masks[prefix], mask)

    return dict(merged_masks)
def split_textures_to_parts(part_masks, extracted_textures):
    split_textures = {}

    for part_name, mask in part_masks.items():
        # 提取部分名称的前缀（例如 "Torso_", "Upper_Arm_Left_"）
        prefix = '_'.join(part_name.split('_')[:-1])
        # 从 extracted_textures 中获取对应的合并后的纹理
        texture = extracted_textures.get(prefix, None)
        print(prefix)
        if texture is not None:
            # 只保留原始掩码部分的区域
            split_textures[part_name] = texture
        else:
            print("is null")

    return split_textures

def cluster_colors(image, top_left, bottom_right, k=3):
    # 提取正方形区域
    sub_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    # 形状转换为 (num_pixels, 3) 的二维数组
    pixels = sub_image.reshape(-1, 3)

    if len(pixels) != 0:
        # 使用 K-means 聚类
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        new_colors = kmeans.cluster_centers_[kmeans.labels_]
        # 将聚类结果转换回原始形状
        clustered_image = new_colors.reshape(sub_image.shape).astype(np.uint8)

        # 将聚类后的图像放回原图
        image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = clustered_image
        return image, True
    return image,False


def smooth_color_distribution(image, top_left, bottom_right):
    # 提取正方形区域
    sub_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    # 检查是否为空
    if sub_image.size == 0:
        return image,False

    # 对区域进行均值平滑
    smoothed_sub_image = cv2.blur(sub_image, (5, 5))  # 使用5x5的卷积核

    # 将平滑后的图像放回原图
    image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = smoothed_sub_image
    return image,True
def feather_edges(transparent_image, feather_radius=50):
    # 获取 Alpha 通道
    alpha = transparent_image[:, :, 3]

    # 创建羽化掩码
    kernel_size = 2 * feather_radius + 1
    blurred_alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), feather_radius)

    # 使用羽化后的 Alpha 通道更新透明图像
    transparent_image[:, :, 3] = blurred_alpha
    return transparent_image

def smooth_texture_edges(image, blur_radius=5):
    # 对整个图像应用高斯模糊
    return cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
def liquify_image(image, intensity=5.0):
    rows, cols = image.shape[:2]
    src_cols = np.linspace(0, cols - 1, cols)
    src_rows = np.linspace(0, rows - 1, rows)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)

    # 扭曲的强度
    dst_cols = src_cols + intensity * np.sin(src_cols / 10.0)
    dst_rows = src_rows + intensity * np.cos(src_rows / 10.0)

    # 限制映射坐标在有效范围内
    dst_cols = np.clip(dst_cols, 0, cols - 1)
    dst_rows = np.clip(dst_rows, 0, rows - 1)

    # 映射到扭曲后的坐标
    map_x = np.float32(dst_cols)
    map_y = np.float32(dst_rows)

    # 应用映射变换
    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped_image


def extract_texture(image, part_masks, main_mask,min_YCrCb,max_YCrCb):
    merge_part_masks = merge_masks(part_masks)
    main_mask = (main_mask > 0).astype(np.uint8) * 255
    extracted_textures = {}

    for part, mask in merge_part_masks.items():
        mask = (mask > 0).astype(np.uint8) * 255
        mask = mask & ~main_mask
        direction = 'down'
        if 'Torso' in part:
            print(f'{part} is down')
        else:
            direction = 'up'
            print(f'{part} is up')
        # 提取最大正方形区域
        result = extract_max_square(mask, image, direction=direction, min_YCrCb=min_YCrCb, max_YCrCb=max_YCrCb)
        if not result:
            print(f'{part} did not had wen li')
            continue
        (top_left, bottom_right, max_size) = result

        # 提取纹理部分
        extracted_texture = image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, :]
        extracted_mask = mask[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

        # 创建透明图像
        transparent_image = np.zeros((extracted_texture.shape[0], extracted_texture.shape[1], 4), dtype=np.uint8)
        transparent_image[:, :, :3] = extracted_texture
        transparent_image[:, :, 3] = extracted_mask

        final_trans = transparent_image

        # 平滑颜色或聚类
        # 方法2：颜色聚类
        # transparent_image[:, :, :3], had_find= cluster_colors(transparent_image[:, :, :3], top_left, bottom_right, k=3)
        # 方法1：均值平滑
        transparent_image[:, :, :3],had_find = smooth_color_distribution(transparent_image[:, :, :3], top_left, bottom_right)


        # 保存提取的纹理
        if had_find:
            final_trans = transparent_image
        # 应用液化效果
        # final_trans = liquify_image(final_trans, intensity=5.0)

        # 高斯模糊边缘
        final_trans = smooth_texture_edges(final_trans)

        # 羽化边缘
        final_trans = feather_edges(final_trans)

        extracted_textures[part] = final_trans
        print(f'{part} had find wen li')
    return split_textures_to_parts(part_masks, extracted_textures)

def find_non_empty_texture(extracted_textures):
    for part,texture in extracted_textures.items():
        if np.count_nonzero(texture[:, :, 3]) > 0:
            return texture,part
    return None,None

def is_skin_color(pixel, min_YCrCb, max_YCrCb):
    return np.all(pixel >= min_YCrCb) and np.all(pixel <= max_YCrCb)
def warp_texture_t(base_image, target_mask, texture, main_mask):
    target_mask = (target_mask > 0).astype(np.uint8) * 255
    main_mask = (main_mask > 0).astype(np.uint8) * 255
    filled_image = base_image.copy()

    overlap_mask = (target_mask > 0) & (main_mask > 0)
    overlap_y, overlap_x = np.where(overlap_mask)
    if len(overlap_y) == 0 or len(overlap_x) == 0:
        return filled_image

    overlap_x_min, overlap_x_max = overlap_x.min(), overlap_x.max()
    overlap_y_min, overlap_y_max = overlap_y.min(), overlap_y.max()

    texture_mask = texture[:, :, 3] > 0
    texture_y, texture_x = np.where(texture_mask)
    if len(texture_y) == 0 or len(texture_x) == 0:
        return filled_image

    texture_x_min, texture_x_max = texture_x.min(), texture_x.max()
    texture_y_min, texture_y_max = texture_y.min(), texture_y.max()
    texture_width = texture_x_max - texture_x_min + 1
    texture_height = texture_y_max - texture_y_min + 1

    repeats_x = (overlap_x_max - overlap_x_min + 1 + texture_width - 1) // texture_width
    repeats_y = (overlap_y_max - overlap_y_min + 1 + texture_height - 1) // texture_height

    for i in range(repeats_y):
        for j in range(repeats_x):
            top_left_x = overlap_x_min + j * texture_width
            top_left_y = overlap_y_min + i * texture_height
            bottom_right_x = min(top_left_x + texture_width, overlap_x_max + 1)
            bottom_right_y = min(top_left_y + texture_height, overlap_y_max + 1)

            target_x_start = max(top_left_x, overlap_x_min)
            target_y_start = max(top_left_y, overlap_y_min)
            target_x_end = min(bottom_right_x, overlap_x_max + 1)
            target_y_end = min(bottom_right_y, overlap_y_max + 1)

            source_x_start = texture_x_min
            source_y_start = texture_y_min
            source_x_end = source_x_start + (target_x_end - target_x_start)
            source_y_end = source_y_start + (target_y_end - target_y_start)

            region_mask = overlap_mask[target_y_start:target_y_end, target_x_start:target_x_end]
            filled_image[target_y_start:target_y_end, target_x_start:target_x_end, :3] = np.where(
                region_mask[:, :, np.newaxis],
                texture[source_y_start:source_y_end, source_x_start:source_x_end, :3],
                filled_image[target_y_start:target_y_end, target_x_start:target_x_end, :3]
            )

        # 仅对重叠区域应用颜色聚类和均值平滑
        filled_part = filled_image[overlap_y_min:overlap_y_max + 1, overlap_x_min:overlap_x_max + 1]
        filled_part_mask = overlap_mask[overlap_y_min:overlap_y_max + 1, overlap_x_min:overlap_x_max + 1]

        # 仅处理重叠区域的有效部分
        if np.any(filled_part_mask):
            clustered_part = cluster_colors_total(filled_part[filled_part_mask], k=100)
            smoothed_part = smooth_color_distribution_total(clustered_part.reshape(filled_part[filled_part_mask].shape),
                                                      blur_radius=5)
            filled_image[overlap_y_min:overlap_y_max + 1, overlap_x_min:overlap_x_max + 1][
                filled_part_mask] = smoothed_part

        return filled_image
def extract_max_square(mask, image, max_variation=250, direction='up', min_YCrCb=np.array([205, 145, 100]), max_YCrCb=np.array([255, 175, 120]), size_tolerance=5):
    h, w = mask.shape
    dp = np.zeros((h, w), dtype=int)

    max_size = 0
    candidates = []

    # 转换图像到 Lab 颜色空间
    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # 转换图像到 YCrCb 颜色空间
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    for i in range(1, h):
        for j in range(1, w):
            if mask[i, j] > 0:
                if i > 0 and j > 0:
                    # 计算相邻像素的颜色差异
                    current_pixel = image_Lab[i, j]
                    above_pixel = image_Lab[i - 1, j]
                    left_pixel = image_Lab[i, j - 1]
                    top_left_pixel = image_Lab[i - 1, j - 1]

                    variation = max(
                        np.linalg.norm(current_pixel - above_pixel),
                        np.linalg.norm(current_pixel - left_pixel),
                        np.linalg.norm(current_pixel - top_left_pixel)
                    )

                    # 检查当前像素和相邻像素是否都在皮肤颜色范围内
                    current_pixel_YCrCb = image_YCrCb[i, j]
                    above_pixel_YCrCb = image_YCrCb[i - 1, j]
                    left_pixel_YCrCb = image_YCrCb[i, j - 1]
                    top_left_pixel_YCrCb = image_YCrCb[i - 1, j - 1]

                    if (variation <= max_variation and
                        is_skin_color(current_pixel_YCrCb, min_YCrCb, max_YCrCb) and
                        is_skin_color(above_pixel_YCrCb, min_YCrCb, max_YCrCb) and
                        is_skin_color(left_pixel_YCrCb, min_YCrCb, max_YCrCb) and
                        is_skin_color(top_left_pixel_YCrCb, min_YCrCb, max_YCrCb)):
                        dp[i, j] = min(dp[i, j - 1], dp[i - 1, j], dp[i - 1, j - 1]) + 1
                        if dp[i, j] > max_size:
                            max_size = dp[i, j]
                            candidates = [(i, j)]
                        elif max_size - dp[i, j] <= size_tolerance:
                            candidates.append((i, j))
                    else:
                        dp[i, j] = 0
                else:
                    dp[i, j] = 1  # 边界处初始为1
                    if dp[i, j] > max_size:
                        max_size = dp[i, j]
                        candidates = [(i, j)]
                    elif max_size - dp[i, j] <= size_tolerance:
                        candidates.append((i, j))

    if max_size == 0:
        return None

    # 根据指定的方向优先返回特定的区域
    if direction == 'up':
        chosen = min(candidates, key=lambda x: x[0])
    elif direction == 'down':
        chosen = max(candidates, key=lambda x: x[0])
    elif direction == 'left':
        chosen = min(candidates, key=lambda x: x[1])
    elif direction == 'right':
        chosen = max(candidates, key=lambda x: x[1])
    else:
        chosen = candidates[0]  # 默认选择第一个找到的

    top_left = (chosen[0] - max_size + 1, chosen[1] - max_size + 1)
    return (top_left, chosen, max_size)
def cluster_colors_total(image, k=3):
    # 形状转换为 (num_pixels, 3) 的二维数组
    pixels = image.reshape(-1, 3)

    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    new_colors = kmeans.cluster_centers_[kmeans.labels_]

    # 将聚类结果转换回原始形状
    clustered_image = new_colors.reshape(image.shape).astype(np.uint8)
    return clustered_image

def smooth_color_distribution_total(image, blur_radius=5):
    # 对区域进行均值平滑
    smoothed_image = cv2.blur(image, (blur_radius, blur_radius))
    return smoothed_image

def warp_texture(base_image, target_mask, texture, main_mask):
    target_mask = (target_mask > 0).astype(np.uint8) * 255
    main_mask = (main_mask > 0).astype(np.uint8) * 255
    filled_image = base_image.copy()

    overlap_mask = (target_mask > 0) & (main_mask > 0)
    overlap_y, overlap_x = np.where(overlap_mask)
    if len(overlap_y) == 0 or len(overlap_x) == 0:
        return filled_image

    overlap_x_min, overlap_x_max = overlap_x.min(), overlap_x.max()
    overlap_y_min, overlap_y_max = overlap_y.min(), overlap_y.max()

    texture_mask = texture[:, :, 3] > 0
    texture_y, texture_x = np.where(texture_mask)
    if len(texture_y) == 0 or len(texture_x) == 0:
        return filled_image

    texture_x_min, texture_x_max = texture_x.min(), texture_x.max()
    texture_y_min, texture_y_max = texture_y.min(), texture_y.max()
    texture_width = texture_x_max - texture_x_min + 1
    texture_height = texture_y_max - texture_y_min + 1

    repeats_x = (overlap_x_max - overlap_x_min + 1 + texture_width - 1) // texture_width
    repeats_y = (overlap_y_max - overlap_y_min + 1 + texture_height - 1) // texture_height

    for i in range(repeats_y):
        for j in range(repeats_x):
            top_left_x = overlap_x_min + j * texture_width
            top_left_y = overlap_y_min + i * texture_height
            bottom_right_x = min(top_left_x + texture_width, overlap_x_max + 1)
            bottom_right_y = min(top_left_y + texture_height, overlap_y_max + 1)

            target_x_start = max(top_left_x, overlap_x_min)
            target_y_start = max(top_left_y, overlap_y_min)
            target_x_end = min(bottom_right_x, overlap_x_max + 1)
            target_y_end = min(bottom_right_y, overlap_y_max + 1)

            source_x_start = texture_x_min
            source_y_start = texture_y_min
            source_x_end = source_x_start + (target_x_end - target_x_start)
            source_y_end = source_y_start + (target_y_end - target_y_start)

            region_mask = overlap_mask[target_y_start:target_y_end, target_x_start:target_x_end]
            filled_image[target_y_start:target_y_end, target_x_start:target_x_end, :3] = np.where(
                region_mask[:, :, np.newaxis],
                texture[source_y_start:source_y_end, source_x_start:source_x_end, :3],
                filled_image[target_y_start:target_y_end, target_x_start:target_x_end, :3]
            )

    filled_part = filled_image[overlap_y_min:overlap_y_max + 1, overlap_x_min:overlap_x_max + 1]
    # filled_part_clustered = cluster_colors_total(filled_part, k=3)
    # filled_part_smoothed = smooth_color_distribution_total(filled_part_clustered, blur_radius=5)

    filled_image[overlap_y_min:overlap_y_max + 1, overlap_x_min:overlap_x_max + 1] = filled_part

    return filled_image

def resize_and_apply_texture(base_image, target_mask, texture, main_mask, is_texture_rgb=True):
    target_mask = (target_mask > 0).astype(np.uint8) * 255
    main_mask = (main_mask > 0).astype(np.uint8) * 255
    filled_image = base_image.copy()

    # 计算目标区域的边界框
    target_y, target_x = np.where(target_mask > 0)
    if len(target_y) == 0 or len(target_x) == 0:
        return filled_image

    x_min, x_max = target_x.min(), target_x.max()
    y_min, y_max = target_y.min(), target_y.max()
    target_height = y_max - y_min + 1
    target_width = x_max - x_min + 1

    # 提取纹理的有效区域
    texture_mask = texture[:, :, 3]
    texture_y, texture_x = np.where(texture_mask > 0)
    if len(texture_y) == 0 or len(texture_x) == 0:
        return filled_image

    texture_x_min, texture_x_max = texture_x.min(), texture_x.max()
    texture_y_min, texture_y_max = texture_y.min(), texture_y.max()

    if is_texture_rgb:
        texture = cv2.cvtColor(texture, cv2.COLOR_BGRA2BGR)  # 去除Alpha通道，转换为BGR

    # 裁剪和缩放纹理
    cropped_texture = texture[texture_y_min:texture_y_max+1, texture_x_min:texture_x_max+1]
    resized_texture = cv2.resize(cropped_texture, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # 计算同时在 target_mask 和 main_mask 内的区域
    combined_mask = (target_mask[y_min:y_max+1, x_min:x_max+1] > 0) & (main_mask[y_min:y_max+1, x_min:x_max+1] > 0)

    # 应用纹理到目标区域
    for i in range(3):
        filled_image[y_min:y_max+1, x_min:x_max+1, i] = np.where(combined_mask,
                                                                  resized_texture[:, :, i],
                                                                  filled_image[y_min:y_max+1, x_min:x_max+1, i])

    return filled_image


if __name__ == "__main__":

    mask = handlerMask("/Users/dean/Desktop/uploads/95989817d36e2e8_size540_w2500_h3750.jpg")
    # 保存掩码图像
    out_dir = os.path.dirname("/Users/dean/Desktop/uploads/image_densepose_contour6.png")
    if len(out_dir) > 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite("/Users/dean/Desktop/uploads/image_densepose_contour6.png", mask)