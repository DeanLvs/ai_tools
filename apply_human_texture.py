import cv2
import numpy as np
from PIL import Image

def extract_texture(image, part_masks, main_mask):
    """
    从图像中提取多个部位掩码区域的纹理，排除大掩码区域。
    :param image: 输入图像（NumPy 数组）
    :param part_masks: 各部位的掩码字典（NumPy 数组）
    :param main_mask: 大掩码（NumPy 数组）
    :return: 提取的纹理图像字典
    """
    # 确保大掩码是二进制的（0 或 255）
    main_mask = (main_mask > 0).astype(np.uint8) * 255

    extracted_textures = {}

    for part, mask in part_masks.items():
        # 确保部位掩码是二进制的（0 或 255）
        mask = (mask > 0).astype(np.uint8) * 255

        # 去除大掩码内的像素
        mask = mask & ~main_mask

        # 创建一个全零的透明背景图像 (RGBA)
        transparent_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

        # 将图像中的区域应用到透明背景上，保留掩码区域
        for i in range(3):  # 对于每个颜色通道
            transparent_image[:, :, i] = image[:, :, i] * mask

        # 将 alpha 通道设置为掩码值
        transparent_image[:, :, 3] = mask

        # 保存结果到字典
        extracted_textures[part] = transparent_image

    return extracted_textures

def apply_texture(image, mask, texture, mask_A=None):
    """
    将纹理应用于图像的特定掩码区域，但排除 mask_A 指定的区域。
    :param image: 原始图像（NumPy 数组）
    :param mask: 掩码（NumPy 数组）
    :param texture: 纹理图像（NumPy 数组）
    :param mask_A: 要排除的掩码区域（NumPy 数组）
    :return: 应用了纹理的图像
    """
    # 调整纹理图像大小与掩码一致
    texture_resized = cv2.resize(texture, (mask.shape[1], mask.shape[0]))

    # 创建空白图像以存储结果
    textured_image = image.copy()

    # 将 mask_A 转换为布尔数组
    if mask_A is not None:
        exclude_mask = mask_A.astype(bool)
    else:
        exclude_mask = np.zeros(mask.shape, dtype=bool)

    # 使用掩码将纹理图像应用于原始图像，排除 mask_A 区域
    for i in range(3):  # 对于每个颜色通道
        textured_image[:, :, i] = np.where((mask > 0) & (~exclude_mask),
                                           texture_resized[:, :, i],
                                           image[:, :, i])

    return textured_image

def build_human_model_with_texture(image_path, texture_paths, masks, mask_A=None):
    """
    为人体模型应用皮肤纹理。
    :param image_path: 原始图像的路径
    :param texture_paths: 各部位的皮肤纹理图像路径的字典
    :param masks: 部位掩码的字典
    :param mask_A: 要排除的掩码区域（NumPy 数组）
    :return: 应用了纹理的人体模型图像
    """
    # 加载原始图像
    image_pil = Image.open(image_path).convert("RGB")
    image = np.array(image_pil)

    # 应用纹理到各个部位掩码
    textured_image = image.copy()
    for part, mask in masks.items():
        texture_path = texture_paths.get(part)
        if texture_path:
            texture_pil = Image.open(texture_path).convert("RGB")
            texture = np.array(texture_pil)
            textured_image = apply_texture(textured_image, mask, texture, mask_A)

    return textured_image

# 示例用法
image_path = 'path_to_image.jpg'  # 替换为您的原始图像路径

# 各部位的纹理图像路径
texture_paths = {
    'Torso': 'path_to_torso_texture.jpg',        # 替换为躯干的纹理图像路径
    'Right Hand': 'path_to_right_hand_texture.jpg', # 替换为右手的纹理图像路径
    'Left Hand': 'path_to_left_hand_texture.jpg',  # 替换为左手的纹理图像路径
    'Right Leg': 'path_to_right_leg_texture.jpg',  # 替换为右腿的纹理图像路径
    'Left Leg': 'path_to_left_leg_texture.jpg',   # 替换为左腿的纹理图像路径
    'Head': 'path_to_head_texture.jpg',           # 替换为头部的纹理图像路径
    # 其他部位的纹理路径...
}

# 假设 masks 是一个字典，包含每个部位的掩码（NumPy 数组）
masks = {
    'Torso': np.zeros((582, 924), dtype=np.uint8),  # 示例掩码
    'Right Hand': np.zeros((582, 924), dtype=np.uint8),  # 示例掩码
    'Left Hand': np.zeros((582, 924), dtype=np.uint8),  # 示例掩码
    'Right Leg': np.zeros((582, 924), dtype=np.uint8),  # 示例掩码
    'Left Leg': np.zeros((582, 924), dtype=np.uint8),  # 示例掩码
    'Head': np.zeros((582, 924), dtype=np.uint8),  # 示例掩码
    # 其他部位的掩码...
}

# 示例的排除掩码
mask_A = np.zeros((582, 924), dtype=np.uint8)
mask_A[100:200, 100:200] = 255  # 示例，实际使用中根据需求设置

# 调用函数生成应用纹理的人体模型图像
textured_image = build_human_model_with_texture(image_path, texture_paths, masks, mask_A)

# 保存或显示结果
textured_image_pil = Image.fromarray(textured_image)
textured_image_pil.save("output_textured_human.png")
textured_image_pil.show()