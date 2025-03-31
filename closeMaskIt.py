from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import os

def save_binary_clothing_mask(image_path, save_dir="output_masks"):
    """
    生成二值化的衣服掩码图像并保存到指定目录。

    参数:
        image_path (str): 输入图像的路径。
        save_dir (str): 掩码图像的保存目录，默认是 "output_masks"。

    返回:
        str: 保存的掩码图像文件路径。
    """
    # 加载模型和处理器
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    # 读取输入图像
    image = Image.open(image_path)

    # 预处理图像
    inputs = processor(images=image, return_tensors="pt")

    # 模型推理
    outputs = model(**inputs)
    logits = outputs.logits

    # 上采样到输入图像的尺寸
    upsampled_logits = F.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False
    )

    # 获取每个像素的预测类别
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    # 定义衣服类别的标签
    clothing_labels = {4, 5, 6, 7}

    # 生成二值化掩码
    binary_mask = np.isin(pred_seg.numpy(), list(clothing_labels)).astype(np.uint8)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存二值化掩码图像
    save_path = os.path.join(save_dir, f"binary_clothing_mask_{os.path.basename(image_path)}")
    mask_image = Image.fromarray(binary_mask * 255)  # 转换为 0 和 255 图像
    mask_image.save(save_path)

    return save_path