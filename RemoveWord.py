import os
import torch
from downloadC import download_file_c
from CoreService import gen_clear_pic
import time
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from PIL import ImageDraw,Image, ImageFilter
from ImageProcessorSDX import CustomInpaintPipeline
import base64
import numpy as np
import requests
import gc
import cv2
from hezar.models import Model
from hezar.utils import load_image, draw_boxes, show_image

if __name__ == '__main__':
    # 检查 GPU 是否可用
    use_cuda = torch.cuda.is_available()
    print(f"CUDA Available: {use_cuda}")

    # 加载模型到 GPU 或 CPU
    device = "cuda" if use_cuda else "cpu"
    model = Model.load("hezarai/CRAFT", device=device)
    model.to("cuda")  # 手动将模型转移到 GPU
    # 加载图像
    image_path = "feff0d5e01acc7049b529823b77532a8.png"
    image = load_image(image_path)

    # 进行文本检测
    outputs = model.predict(image)

    # 打印 outputs 的类型
    print(f"Type of outputs: {type(outputs)}")

    # 检查 outputs[0] 的类型和内容
    print(f"Type of outputs[0]: {type(outputs[0])}")
    print(f"Content of outputs[0]: {outputs[0]}")

    # 如果 outputs 是列表或字典，可以查看其键或索引
    if isinstance(outputs[0], dict):
        print(f"Keys in outputs[0]: {outputs[0].keys()}")

    # 检查 boxes 的内容
    if "boxes" in outputs[0]:
        print(f"Boxes: {outputs[0]['boxes']}")

    # 绘制检测框并保存图像
    try:
        # 确保 image 是 NumPy 数组
        image_np = np.array(image)

        # 假设 outputs[0]["boxes"] 是包含文字多边形的坐标列表
        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)  # 创建空白掩码
        for box in outputs[0]["boxes"]:
            polygon = np.array(box, dtype=np.int32)  # 转换为多边形坐标
            cv2.fillPoly(mask, [polygon], 255)  # 填充多边形区域，生成掩码

        # 保存掩码
        cv2.imwrite("text_maswwwk.png", mask) # 保存生成的掩码

        result_image = draw_boxes(image, outputs[0]["boxes"])
        result_image.save("output_image_with_boxes.jpg")
        print("Saved result image with boxes successfully.")
    except Exception as e:
        print(f"Error in processing or saving image: {e}")

    # 可选：显示图像
    # show_image(result_image, "text_detected")