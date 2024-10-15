#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from book_yes_logger_config import logger
import traceback
from sklearn.cluster import KMeans

def convert_to_ndarray(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError("Unsupported image type")

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
        logger.info(prefix)
        if texture is not None:
            # 只保留原始掩码部分的区域
            split_textures[part_name] = texture
        else:
            logger.info("is null")

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


def process_part(part, mask, image, main_mask):
    mask = (mask > 0).astype(np.uint8) * 255
    mask = mask & ~main_mask

    result = extract_max_square_fast(mask, image)
    if not result:
        logger.info(f'{part} did not find wen li')
        return part, None

    (top_left, bottom_right, max_size) = result

    extracted_texture = image[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1, :]
    extracted_mask = mask[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]

    transparent_image = np.zeros((extracted_texture.shape[0], extracted_texture.shape[1], 4), dtype=np.uint8)
    transparent_image[:, :, :3] = extracted_texture
    transparent_image[:, :, 3] = extracted_mask

    final_trans = transparent_image

    transparent_image[:, :, :3], had_find = smooth_color_distribution(transparent_image[:, :, :3], top_left,
                                                                      bottom_right)

    if had_find:
        final_trans = transparent_image

    final_trans = smooth_texture_edges(final_trans)
    final_trans = feather_edges(final_trans)

    logger.info(f'{part} found wen li')
    return part, final_trans

def extract_texture(image, part_masks, main_mask ,min_YCrCb=None, max_YCrCb=None):
    merge_part_masks = merge_masks(part_masks)
    main_mask = (main_mask > 0).astype(np.uint8) * 255
    extracted_textures = {}

    # 使用 ThreadPoolExecutor 并行处理每个部分
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []  # 创建一个空的字典来保存 future 和 part 的映射关系
        for part, mask in merge_part_masks.items():
            future = executor.submit(process_part, part, mask, image, main_mask)
            futures.append(future)  # 将 future 和 part 绑定在一起，便于后续查找

    for future in concurrent.futures.as_completed(futures):
        try:
            part, result = future.result()
            extracted_textures[part] = result
        except Exception as exc:
            logger.info(f'{part} generated an exception: {exc}')
    return split_textures_to_parts(part_masks, extracted_textures)

def extract_texture_b(image, part_masks, main_mask ,min_YCrCb=None, max_YCrCb=None):
    merge_part_masks = merge_masks(part_masks)
    main_mask = (main_mask > 0).astype(np.uint8) * 255
    extracted_textures = {}

    for part, mask in merge_part_masks.items():
        mask = (mask > 0).astype(np.uint8) * 255
        mask = mask & ~main_mask
        # 提取最大正方形区域
        result = extract_max_square_fast(mask, image)
        if not result:
            logger.info(f'{part} did not had wen li')
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
        logger.info(f'{part} had find wen li')
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
def process_block(i, j, integral_hue, integral_sat, block_size, hue_peak, sat_peak):
    block_hue_sum = integral_hue[i + block_size, j + block_size] - integral_hue[i, j + block_size] - integral_hue[i + block_size, j] + integral_hue[i, j]
    block_sat_sum = integral_sat[i + block_size, j + block_size] - integral_sat[i, j + block_size] - integral_sat[i + block_size, j] + integral_sat[i, j]

    avg_hue = block_hue_sum / (block_size ** 2)
    avg_sat = block_sat_sum / (block_size ** 2)

    hue_diff = np.abs(avg_hue - hue_peak)
    sat_diff = np.abs(avg_sat - sat_peak)
    variation = hue_diff + sat_diff

    return variation, (i, j)


def extract_max_square_fast(mask, image, block_size=5, step_size=4):
    logger.info(f'start extract_max_square...')
    valid_pixels = np.sum(mask > 0)
    min_width = max(int(np.sqrt(valid_pixels * 0.3)), block_size)
    min_height = max(int(np.sqrt(valid_pixels * 0.3)), block_size)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
    upper_hsv = np.array([20, 150, 255], dtype=np.uint8)

    hsv_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    lower_ycrcb = np.array([87, 138, 107], dtype=np.uint8)
    upper_ycrcb = np.array([255, 146, 128], dtype=np.uint8)

    ycrcb_mask = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)

    combined_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    combined_mask = cv2.bitwise_and(combined_mask, mask)

    masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=combined_mask)
    integral_hue = cv2.integral(masked_image[:, :, 0])
    integral_sat = cv2.integral(masked_image[:, :, 1])

    hist = cv2.calcHist([masked_image], [0, 1], combined_mask, [180, 256], [0, 180, 0, 256])

    hue_peak, sat_peak = np.unravel_index(np.argmax(hist), hist.shape)

    h, w = mask.shape

    if h < min_height or w < min_width:
        fill_block = np.full((block_size, block_size, 3), [hue_peak, sat_peak, 255], dtype=np.uint8)
        logger.info("Image or mask is too small, filling with dominant color.")
        return ((0, 0), (block_size - 1, block_size - 1), block_size)

    min_variation = float('inf')
    max_square = None
    logger.info(f'start ThreadPoolExecutor extract_max_square...')
    # 提交任务并并行处理
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(process_block, i, j, integral_hue, integral_sat, block_size, hue_peak, sat_peak)
            for i in range(0, h - block_size + 1, step_size)
            for j in range(0, w - block_size + 1, step_size)
        ]

    # 在主线程中收集结果
    for future in futures:
        variation, (i, j) = future.result()
        if variation < min_variation:
            min_variation = variation
            max_square = ((i, j), (i + block_size - 1, j + block_size - 1))

    if max_square is None:
        return None

    logger.info(f'finish extract_max_square_fast...')
    return (max_square[0], max_square[1], block_size)

def extract_max_square(mask, image, max_variation=250, direction='up', min_YCrCb=np.array([205, 145, 100]), max_YCrCb=np.array([255, 175, 120]), size_tolerance=5):
    logger.info(f'start extract_max_square...')
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
    logger.info(f'finish extract_max_square...')
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
