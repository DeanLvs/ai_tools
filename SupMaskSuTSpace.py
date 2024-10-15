#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
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
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
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
    _lock = threading.Lock()

    def __new__(cls, config_file, model_file, device="cpu"):
        with cls._lock:  # 确保线程安全的单例创建
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
        cfg.MODEL.KEYPOINT_ON = True
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

        return masks['all_body']

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

    def generate_key_point(self, image):

        with torch.no_grad():
            outputs = self.predictor(image)

        # 创建一个纯黑色的背景图像
        black_background = np.zeros(image.shape, dtype=np.uint8)
        from detectron2.utils.visualizer import Visualizer, ColorMode
        # 移除边框数据
        instances = outputs["instances"].to("cpu")
        if instances.has("pred_boxes"):
            instances.remove("pred_boxes")

        # 使用 Visualizer 绘制关键点和骨架，不绘制边框
        v = Visualizer(black_background[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=3.0,
                       instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(instances)

        # 获取绘制的图像
        result_image = v.get_image()[:, :, ::-1]

        return result_image

def handlerMaskAll(img_path):
    img = read_image(img_path, format="BGR")
    # 配置文件和模型文件路径
    config_file = "detectron2/model_zoo/configs/densepose_rcnn_R_101_FPN_DL_WC2M_s1x.yaml"
    model_file = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2M_s1x/216245790/model_final_de6e7a.pkl"

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
        "exclude": [3,4,5,6,23,24],
        "all_body": [1, 2],
    }
    mask = densepose_model.generate_m_mask(img, parts_to_include=parts_to_include)
    part_mask = mask["all_body"]
    print(f'{part_mask}')
    # 创建一个彩色图像，与掩码图像大小相同，初始值全为 0
    color_image = np.zeros((part_mask.shape[0], part_mask.shape[1], 3), dtype=np.uint8)

    # 定义浅蓝色和浅黄色的 RGB 颜色
    light_blue = [255,165,0]  # 浅蓝色 (R, G, B)
    light_yellow = [255,235,205]  # 浅黄色 (R, G, B)

    # 将掩码中的黑色 (0) 替换为浅蓝色
    color_image[part_mask == 0] = light_blue

    # 将掩码中的白色 (1) 替换为浅黄色
    color_image[part_mask == 255] = light_yellow
    mask["all_body"] = color_image
    return mask
def handlerMaskWC(img_path):
    img = read_image(img_path, format="BGR")
    # 配置文件和模型文件路径
    # config_file = "detectron2/model_zoo/configs/densepose_rcnn_R_101_FPN_DL_WC2M_s1x.yaml"
    # model_file = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2M_s1x/216245790/model_final_de6e7a.pkl"
    # config_file = "detectron2/model_zoo/configs/densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml"
    # model_file = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2_s1x/173294801/model_final_6e1ed1.pkl"
    config_file = "detectron2/model_zoo/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
    model_file = "https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl"

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
    # parts_to_include = [3,4,5,6,23,24]
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
        "exclude": [3, 4, 5, 6, 23, 24],
        "all_body": [1, 2],
    }
    mask = densepose_model.generate_key_point(img)

    # handlerMaskWC_UU(img_path)
    return mask


def handlerMaskWCKey(img_path):
    img = read_image(img_path, format="BGR")
    # 配置文件和模型文件路径
    # config_file = "detectron2/model_zoo/configs/densepose_rcnn_R_101_FPN_DL_WC2M_s1x.yaml"
    # model_file = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2M_s1x/216245790/model_final_de6e7a.pkl"
    # config_file = "detectron2/model_zoo/configs/densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml"
    # model_file = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2_s1x/173294801/model_final_6e1ed1.pkl"
    config_file = "detectron2/model_zoo/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
    model_file = "https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl"

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
    # parts_to_include = [3,4,5,6,23,24]
    # 加载图像

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = densepose_model.generate_key_point(image)

    # handlerMaskWC_UU(img_path)
    return mask


if __name__ == "__main__":
    parts_to_include = [1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    mask = handlerMaskWC("/Users/dean/Desktop/uploads/processed_588cb120a2ac4104a231101b20831e8f\ \(2\).jpeg")
    # 保存掩码图像
    out_dir = os.path.dirname("/Users/dean/Desktop/uploads/image_densessspose_contour6.png")
    if len(out_dir) > 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite("/Users/dean/Desktop/uploads/image_densepose_contour6.png", mask)