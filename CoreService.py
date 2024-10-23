import os, gc, uuid
from telegram import InputMediaPhoto
from ImageProcessorSDXCN import CustomInpaintPipeline as sdxlInpainting
from BookYesCommon import GenContextParam, GenSavePathParam, process_req_data, get_rest_key, translate_baidu
import numpy as np
import hashlib
import traceback
import subprocess
import torch
import scipy.ndimage
from PIL import ImageDraw,Image, ImageFilter
from concurrent.futures import ThreadPoolExecutor
import cv2, base64
from SupMaskSu import extract_texture, warp_texture, find_non_empty_texture
import concurrent.futures
import requests, io
import pickle
import zipfile
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
# process_a.py
from book_yes_logger_config import logger

main_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

main_executor_control = concurrent.futures.ThreadPoolExecutor(max_workers=16)

sdxl_ac_in_cpu = None
sdxl_ac_in_gpu = None

def save_image_as_binary(image: Image.Image, file_path: str):
    """
    将 PIL 图像转换为二进制数据流并保存到指定文件
    :param image: PIL 图像对象
    :param file_path: 要保存的二进制文件路径
    """
    # 将图像转换为 NumPy 数组
    image_array = np.array(image)

    # 将 NumPy 数组转换为二进制数据
    binary_data = image_array.tobytes()

    # 将二进制数据写入文件
    with open(file_path, 'wb') as binary_file:
        binary_file.write(binary_data)
    logger.info(f"图像数据已成功保存为二进制文件：{file_path}")

def get_def_extracted_textures_file(glob_parts_to_include=None):
    if glob_parts_to_include:
        return glob_parts_to_include
    texture = None
    parts_to_include = {
        "Torso_1": texture,
        "Torso_2": texture,
        "Upper_Leg_Right_7": texture,
        "Upper_Leg_Left_8": texture,
        "Upper_Leg_Right_9": texture,
        "Upper_Leg_Left_10": texture,
        "Lower_Leg_Right_11": texture,
        "Lower_Leg_Left_12": texture,
        "Lower_Leg_Right_13": texture,
        "Lower_Leg_Left_14": texture,
        "Upper_Arm_Left_15": texture,
        "Upper_Arm_Right_16": texture,
        "Upper_Arm_Left_17": texture,
        "Upper_Arm_Right_18": texture,
        "Lower_Arm_Left_19": texture,
        "Lower_Arm_Right_20": texture,
        "Lower_Arm_Left_21": texture,
        "Lower_Arm_Right_22": texture,
    }
    for key in parts_to_include.keys():
        # 读取图片
        skin_texture = cv2.imread(key+'.png', cv2.IMREAD_UNCHANGED)

        # 确保图片是BGRA格式，如果不是，你可能需要转换
        if skin_texture.shape[2] == 3:  # 如果图片是BGR格式的
            skin_texture = cv2.cvtColor(skin_texture, cv2.COLOR_BGR2BGRA)

        # 转换BGRA到RGBA
        parts_to_include[key] = cv2.cvtColor(skin_texture, cv2.COLOR_BGRA2RGBA)
    glob_parts_to_include = parts_to_include
    return parts_to_include

def skin_n_n(pixel):
    # 创建一个 9x9 的数组，数组中的每个点都是 pixel
    skin_texture = np.array([[pixel] * 9] * 9, dtype=np.uint8)

    return cv2.cvtColor(skin_texture, cv2.COLOR_BGRA2RGBA)

def get_def_extracted_textures():
    # 定义一个点 [255, 240, 230, 255]
    parts_to_include = {
        "Torso_1": skin_n_n([236,179,130, 255]),
        "Torso_2": skin_n_n([236,179,130, 255]),
        "Upper_Leg_Right_7": skin_n_n([230, 215, 200, 255]),
        "Upper_Leg_Left_8": skin_n_n([255, 230, 213, 255]),
        "Upper_Leg_Right_9": skin_n_n([250,230, 205, 255]),
        "Upper_Leg_Left_10": skin_n_n([245,218,196, 255]),
        "Lower_Leg_Right_11": skin_n_n([255,218,185, 255]),
        "Lower_Leg_Left_12": skin_n_n([255,222,173, 255]),
        "Lower_Leg_Right_13": skin_n_n([255,228,181, 255]),
        "Lower_Leg_Left_14": skin_n_n([255,248,220, 255]),
        "Upper_Arm_Left_15": skin_n_n([222,176,158, 255]),
        "Upper_Arm_Right_16": skin_n_n([222,176,158, 255]),
        "Upper_Arm_Left_17": skin_n_n([222,176,158, 255]),
        "Upper_Arm_Right_18": skin_n_n([222,176,158, 255]),
        "Lower_Arm_Left_19": skin_n_n([222,186,148, 255]),
        "Lower_Arm_Right_20": skin_n_n([222,186,158, 255]),
        "Lower_Arm_Left_21": skin_n_n([222,186,158, 255]),
        "Lower_Arm_Right_22": skin_n_n([237, 198, 160, 255]),
    }
    return parts_to_include

def generate_unique_key(*args):
    # Concatenate all arguments to create a unique string
    unique_string = '_'.join(map(str, args))
    # Create a hash of the unique string
    return hashlib.md5(unique_string.encode()).hexdigest()

def dilate_mask_np(mask, iterations):
    kernel = np.ones((3, 3), np.uint8)
    fix_mask_dilated = cv2.dilate(mask, kernel, iterations=iterations)
    # 确保 fix_mask 是一个值在 0 和 255 之间的 uint8 类型数组
    fix_mask_dilated = np.clip(fix_mask_dilated, 0, 255).astype(np.uint8)
    return fix_mask_dilated

def dilate_mask_by_percentage(mask, percentage, max_iterations=300, step_size=2):
    """
    根据整体像素的百分比扩张掩码。
    :param mask: 输入掩码（NumPy 数组）
    :param percentage: 需要扩张的百分比（0-100）
    :param max_iterations: 扩张的最大迭代次数（默认100）
    :return: 扩张后的掩码
    """
    # 计算目标扩张面积
    total_pixels = mask.size
    target_pixels = total_pixels * (percentage / 100.0)

    # 初始掩码的非零像素数量
    initial_non_zero = np.count_nonzero(mask)

    # 计算每次迭代后的扩张像素数量
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = mask.copy()
    last_non_zero = initial_non_zero

    for i in range(max_iterations):
        # 扩张掩码
        dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=step_size)
        current_non_zero = np.count_nonzero(dilated_mask)

        # 检查扩张是否达到目标像素数量
        if current_non_zero - initial_non_zero >= target_pixels:
            break

        last_non_zero = current_non_zero

    return np.clip(dilated_mask, 0, 255).astype(np.uint8)


def dilate_mask(mask, iterations):
    mask_array = np.array(mask)
    dilated_mask = scipy.ndimage.binary_dilation(mask_array, iterations=iterations)
    dilated_mask = (dilated_mask * 255).astype(np.uint8)
    return Image.fromarray(dilated_mask)

def mask_to_pixel_positions(mask):
    # 使用 np.nonzero 获取掩码中非零值的坐标
    y_indices, x_indices = np.nonzero(mask)
    # 将 y_indices 和 x_indices 转换成坐标对列表
    pixel_positions = list(zip(x_indices, y_indices))
    return pixel_positions
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

def remove_exclude_mask(mask_clear_ndarray, exclude_mask):
    # 确保 exclude_mask 的大小与 mask_clear_ndarray 一致
    if mask_clear_ndarray.shape != exclude_mask.shape:
        raise ValueError("The shapes of mask_clear and exclude_mask do not match.")

    # 从 mask_clear_ndarray 中去除 exclude_mask 的区域
    result_ndarray = np.where(exclude_mask > 0, 0, mask_clear_ndarray)

    return result_ndarray

def free_dense_req():
    url = "http://localhost:9080/"
    requests.get(url + 're')
def dense_req(file_path, free_fast):
    url = "http://localhost:9080/"
    data = {'file_path': file_path}
    # 发起请求
    response = requests.post(url+'inpaint', data=data)

    # 检查响应状态
    if response.status_code == 200:
        try:
            zip_data = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_data, 'r') as zf:
                # 读取 d_pose_resized 和 masks
                d_pose_resized = zf.read("d_pose_resized.png")
                d_3d_canny_resized = zf.read("d_3d_canny_resized.png")
                masks = zf.read("masks.pkl")
                return d_pose_resized, pickle.loads(masks), d_3d_canny_resized
        except Exception as e:
            logger.info(f"Error processing response: {e}")
            return None, None, None
        finally:
            if free_fast:
                free_dense_req()
    else:
        logger.info(f"Request failed with status code {response.status_code}")
        return None, None, None
def free_detect_control_image():
    url = 'http://localhost:5060/'
    requests.get(url + 're')
def free_generate_normal_map():
    url = 'http://localhost:5070/'
    requests.get(url + 're')

def gen_mask_f(file_path_e, file_i_e, free_fast_e=True):

    def try_mask_it(file_path, image_pil, free_fast):
        logger.info(f"start gen org close mask")
        # remove_and_rege_mask 放大后的衣服区域
        remove_and_rege_mask, close_mask_img, not_close_mask_img, segmentation_image_pil = re_auto_mask(file_path, free_fast)
        merged_mask_pixels = np.argwhere(remove_and_rege_mask > 0)
        merged_mask_pixels = [[int(x), int(y)] for y, x in merged_mask_pixels]
        mask_clear = Image.new("L", image_pil.size, 0)
        draw = ImageDraw.Draw(mask_clear)
        for point in merged_mask_pixels:
            draw.point(point, fill=255)
        logger.info(f"finish gen org close mask")
        return remove_and_rege_mask, close_mask_img, not_close_mask_img, segmentation_image_pil, mask_clear

    def process_human_mask(file_path, file_i, free_fast):
        logger.info(f'start gen human pose and 3d info....')
        d_pose_resized, humanMask, d_3d_canny_resized = dense_req(file_path, free_fast)
        exclude_mask = humanMask['exclude']
        all_body_mask = humanMask['all_body']
        # 删除键
        if 'exclude' in humanMask:
            del humanMask['exclude']
        if 'all_body' in humanMask:
            del humanMask['all_body']
        output_line_image_pil = Image.open(io.BytesIO(d_3d_canny_resized))
        if 'Torso_line' in humanMask:
            # torso_line = humanMask['Torso_line']
            # left_line = humanMask['left_line']
            # right_line = humanMask['right_line']
            # np.save('torso_line.npy', torso_line)
            # np.save('left_line.npy', left_line)
            # np.save('right_line.npy', right_line)
            # # 创建一个空的输出图像，大小与 torso_line 相同
            # output_line_image = np.zeros_like(torso_line)
            del humanMask['Torso_line']
            if 'left_line' in humanMask:
                del humanMask['left_line']
            if 'right_line' in humanMask:
                del humanMask['right_line']
            # # 使用 Canny 边缘检测器处理 torso_line
            # edges_torso = cv2.Canny(torso_line.astype(np.uint8), 100, 200)
            #
            # # 使用 Canny 边缘检测器处理 left_line
            # edges_left = cv2.Canny(left_line.astype(np.uint8), 100, 200)
            #
            # # 使用 Canny 边缘检测器处理 right_line
            # edges_right = cv2.Canny(right_line.astype(np.uint8), 100, 200)
            #
            # # 逐个叠加边缘到输出图像
            # output_line_image = np.maximum(output_line_image, edges_torso)
            # output_line_image = np.maximum(output_line_image, edges_left)
            # output_line_image = np.maximum(output_line_image, edges_right)
            #
            # # 将输出转换为 PIL 图像
            # output_line_image_pil = d_3d_canny_resized
        logger.info(f'finish gen human pose and 3d info{exclude_mask}')
        return humanMask, output_line_image_pil, d_pose_resized, exclude_mask

    # 姿势图
    def detect_control_image(image_pil, free_fast):
        logger.info(f'start gen pose control....')
        # 获取图像的原始格式
        img_format = image_pil.format if image_pil.format else 'PNG'
        # 将 PIL.Image 对象转换为字节流
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format=img_format)  # 使用原始图像格式
        img_byte_arr.seek(0)  # 将流的指针移动到开头
        # 以二进制形式发送图像数据
        files = {'file_i': img_byte_arr}
        # 发起 POST 请求
        url = 'http://localhost:5060/'
        response = requests.post(url+"inpaint", timeout=99999, files=files)
        # 检查响应状态
        if response.status_code == 200:
            try:
                control_image = pickle.loads(response.content)
                # 返回结果图像
                logger.info(f'finish gen pose control....')
                return control_image
            except Exception as e:
                logger.info(f"gen pose control Error processing response: {e}")
                return None
            finally:
                if free_fast:
                    free_detect_control_image()
        else:
            logger.info(f"gen pose control Request failed with status code {response.status_code}")
            return None
    # 阴影图
    def generate_normal_map(image_pil, next_image_pil, free_fast):
        logger.info(f'start gen depth control....')
        # 获取图像的原始格式
        # 检查 image_pil 是否为 bytes 对象，如果是则转换为 PIL.Image 对象

        if isinstance(image_pil, bytes):
            image_pil = Image.open(io.BytesIO(image_pil))
        img_format = image_pil.format if image_pil.format else 'PNG'
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format=img_format)
        img_byte_arr.seek(0)
        next_img_format = next_image_pil.format if next_image_pil.format else 'PNG'
        # 将 PIL.Image 对象转换为字节流
        next_img_byte_arr = io.BytesIO()
        next_image_pil.save(next_img_byte_arr, format=next_img_format)  # 使用原始图像格式
        next_img_byte_arr.seek(0)  # 将流的指针移动到开头
        # 以二进制形式发送图像数据 ,
        files = {'file_next':next_img_byte_arr, 'file_i': img_byte_arr}
        # 发起 POST 请求
        url = 'http://localhost:5070/'
        response = requests.post(url+"inpaint", timeout=99999, files=files)
        # 检查响应状态
        if response.status_code == 200:
            try:
                zip_data = io.BytesIO(response.content)
                with zipfile.ZipFile(zip_data, 'r') as zf:
                    file_names = zf.namelist()
                    if "depth_3d.png" in file_names:
                        depth_3d = zf.read("depth_3d.png")
                        depth_3d_img = Image.open(io.BytesIO(depth_3d))
                    else:
                        depth_3d_img = None
                    # 读取 d_pose_resized 和 masks
                    depth_org = zf.read("depth_org.png")
                    # 将字节流转换为 PIL.Image 对象
                    depth_org_img = Image.open(io.BytesIO(depth_org))
                    logger.info(f'finish gen depth control....')
                    return depth_org_img, depth_3d_img
            except Exception as e:
                logger.info(f"gen depth Error processing response: {e}")
                return None, None
            finally:
                if free_fast:
                    free_generate_normal_map()
        else:
            logger.info(f"gen depth Request failed with status code {response.status_code}")
            return None, None

    try_mask_it_f = main_executor_control.submit(try_mask_it, file_path_e, file_i_e, free_fast_e)
    detect_control_image_f = main_executor_control.submit(detect_control_image, file_i_e, free_fast_e)

    process_human_mask_f = main_executor_control.submit(process_human_mask, file_path_e, file_i_e, free_fast_e)

    humanMask, f_output_line_image_pil, d_pose_resized_r, exclude_mask = process_human_mask_f.result()
    # 使用3d图出深度图
    generate_normal_map_f = main_executor_control.submit(generate_normal_map, d_pose_resized_r, file_i_e, free_fast_e)
    remove_and_rege_mask, close_mask_img, not_close_mask_img, segmentation_image_pil, mask_clear = try_mask_it_f.result()
    # 剔除掉 手 脚
    # 将 mask_clear 转换为 NumPy 数组
    mask_i_ndarray = np.array(mask_clear)
    # 去除手 头 脚等细节 掩码，保证清理和生成时 不 处理这些细节区域
    clear_mask_array = remove_exclude_mask(mask_i_ndarray, exclude_mask)
    # 移初非衣服区域
    clear_mask_array_final = remove_exclude_mask(clear_mask_array, not_close_mask_img)
    # 将结果转换回 PIL 图像 需要清理的 掩码
    clear_mask = Image.fromarray(clear_mask_array_final.astype(np.uint8))
    # with_torso_mask_array = merge_masks(clear_mask_array_final, all_body_mask)
    with_torso_mask_array = clear_mask_array_final
    # 连着躯干的掩码 暂时没用
    with_torso_mask = Image.fromarray(with_torso_mask_array.astype(np.uint8))

    control_image = detect_control_image_f.result()

    normal_map_img, normal_3d_map_img  = generate_normal_map_f.result()

    return clear_mask, humanMask, control_image, normal_map_img, normal_3d_map_img, with_torso_mask, f_output_line_image_pil, segmentation_image_pil

def apply_skin_color(image_pil, mask):
    skin_color = (172, 219, 255)
    """
    将掩码范围内的像素替换为皮肤颜色。

    参数:
    - image_pil: 原始图片的 PIL.Image 对象。
    - mask: 掩码图片的 NumPy 数组，0 表示不改变的像素，255 表示需要替换为皮肤颜色的区域。
    - skin_color: 要替换的皮肤颜色，默认为 (210, 180, 140) (BGR格式)。

    返回值:
    - 处理后的图片 PIL.Image 对象。
    """
    # 将 PIL 图片转换为 NumPy 数组并转为 BGR 格式，方便使用 OpenCV
    image = np.array(image_pil)[:, :, ::-1]  # 转换为 BGR

    # 确保掩码是二值化的 (0 和 255)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # 将皮肤颜色生成一个与输入图像相同大小的矩阵
    skin_image = np.full(image.shape, skin_color, dtype=np.uint8)

    # 使用掩码将图像中的指定区域替换为皮肤颜色
    result_image = np.where(mask[:, :, None] == 255, skin_image, image)

    # 将结果转换回 RGB 格式并转换为 PIL 图片
    result_image_rgb = result_image[:, :, ::-1]  # 转换为 RGB
    result_image_pil = Image.fromarray(result_image_rgb)

    return result_image_pil

def free_flux_text_gen():
    url = "http://localhost:1006/"
    requests.get(url + 're')

def req_flux_get_face(file_path, prompt, free_fast=True, only_face_path='',seed=2023, gen_type=''):
    # 发起 HTTP POST 请求
    url = "http://localhost:1006/"
    data = {
        'file_path': file_path,
        'prompt': prompt,
        'only_file_path': only_face_path,
        'seed': seed,
        'gen_type': gen_type
    }
    response = requests.post(url+'process_image_mul', data=data)
    # 处理响应的图像
    if response.status_code == 200:
        try:
            # 处理 ZIP 文件
            zip_io = io.BytesIO(response.content)
            images = []

            # 解压 ZIP 文件并加载每张图片
            with zipfile.ZipFile(zip_io, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    with zip_ref.open(file_name) as file:
                        img = Image.open(file)

                        # 将图像加载到内存
                        img_copy = img.copy()
                        images.append(img_copy)

            logger.info(f'-----suc text gen flux plus with multiple images---------')
            return images  # 返回图像列表

        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return None
        finally:
            if free_fast:
                free_flux_text_gen()

    logger.error(f'-----error return suc text gen flux plus---------')
    return None

def free_text_gen(port):
    url = f"http://localhost:{port}/"
    requests.get(url + 're')

def req_get_face(file_path):
    # 发起 HTTP POST 请求
    url = "http://localhost:1060/"
    data = {
        'file_path': file_path
    }
    response = requests.post(url+'face', data=data)
    # 处理响应的图像
    if response.status_code == 200:
        try:
            # 处理 ZIP 文件
            zip_io = io.BytesIO(response.content)
            images = []

            # 解压 ZIP 文件并加载每张图片
            with zipfile.ZipFile(zip_io, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    with zip_ref.open(file_name) as file:
                        img = Image.open(file)

                        # 将图像加载到内存
                        img_copy = img.copy()
                        images.append(img_copy)

            logger.info(f'-----suc text gen sdxl plus with multiple images---------')
            return images  # 返回图像列表

        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return None

    logger.error(f'-----error return suc text gen sdxl plus---------')
    return None

def free_req_replace_face(port):
    url = f"http://localhost:{port}/"
    requests.get(url + 're')
# 5000 1005
def req_process_video(pic_b='', pic_save=None, source_path_list=None, target_path_list=None, fast_free=True, port=0):
    # 定义请求的 URL
    url = f"http://localhost:{port}/process_video_mul"

    # 模拟需要发送的图片路径和其他参数
    data = {
        'pic_b': pic_b,
        'source_path_list': source_path_list,
        'target_path_list': target_path_list,
        'pic_save': pic_save  # 这个是可选的
    }
    logger.info(f'start req replace video face {data}')
    # 发起 POST 请求
    response = requests.post(url, json=data)

    # 检查响应状态码
    if response.status_code == 200:
        # 解析返回的 JSON 响应
        result = response.json()
        if result.get('status') == 'success':
            video_path = result.get('video_path')
            print(f"生成的视频路径: {video_path}")
            if fast_free:
                free_req_replace_face(port)
            return video_path
        else:
            print(f"处理失败，错误信息: {result.get('message')}")
    else:
        print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")

    return None
# 5000 1005
def req_replace_face(pic_b='', pic_save=None, source_path_list=None, target_path_list=None, fast_free=True, port=0):
    try:
        # 定义请求的 URL
        url = f"http://localhost:{port}/process_image_mul"
        # 模拟需要发送的图片路径和其他参数
        data = {
            'pic_b': pic_b,
            'source_path_list': source_path_list,
            'target_path_list': target_path_list,
            'pic_save': pic_save  # 这个是可选的
        }
        logger.info(f'start req replace face {data}')
        # 发起 POST 请求
        response = requests.post(url, json=data)

        # 检查响应状态码
        if response.status_code == 200:
            # 接收返回的 ZIP 文件
            zip_data = io.BytesIO(response.content)

            # 解压 ZIP 文件到本地路径
            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                with zip_ref.open('final_img.png') as file:
                    img = Image.open(file)
                    # 将图像加载到内存
                    img_copy = img.copy()
                    if fast_free:
                        free_req_replace_face(port)
                return img_copy
        else:
            print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
        return None
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        return None
def req_text_gen(file_path, prompt, free_fast=True, only_face_path='',seed=2023, gen_type='', port=1060):
    # 发起 HTTP POST 请求
    url = f"http://localhost:{port}/"
    data = {
        'file_path': file_path,
        'prompt': prompt,
        'only_file_path':only_face_path,
        'seed': seed,
        'gen_type':gen_type
    }
    response = requests.post(url+'inpaint', data=data)
    # 处理响应的图像
    if response.status_code == 200:
        try:
            # 处理 ZIP 文件
            zip_io = io.BytesIO(response.content)
            images = []

            # 解压 ZIP 文件并加载每张图片
            with zipfile.ZipFile(zip_io, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    with zip_ref.open(file_name) as file:
                        img = Image.open(file)

                        # 将图像加载到内存
                        img_copy = img.copy()
                        images.append(img_copy)

            logger.info(f'-----suc text gen sdxl plus with multiple images---------')
            return images  # 返回图像列表

        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return None

        finally:
            if free_fast:
                free_text_gen(port)

    logger.error(f'-----error return suc text gen sdxl plus---------')
    return None

def free_req_lama():
    url = "http://localhost:9090/"
    requests.get(url + 're')
def req_lama(file_path, mask_clear, free_fast):
    # 将 PIL.Image 对象转换为字节流
    img_io = io.BytesIO()
    mask_clear.save(img_io, format='PNG')  # 你可以选择其他格式
    img_io.seek(0)  # 将文件指针重置到开始位置

    # 发起 HTTP POST 请求
    url = "http://localhost:9090/"
    files = {'mask_clear': ('mask.png', img_io, 'image/png')}
    data = {
        'file_path': file_path
    }
    response = requests.post(url+'inpaint', files=files, data=data)

    # 处理响应的图像
    if response.status_code == 200:
        try:
            img_io_re = io.BytesIO(response.content)
            img_pil = Image.open(img_io_re)
            logger.info(f'-----suc return lam---------')
            return np.array(img_pil), True
        except Exception as e:
            logger.info(f"Error processing response: {e}")
            return None, False
        finally:
            if free_fast:
                free_req_lama()
    logger.info(f'-----error return lam---------')
    return None, False

def gen_fix_pic(file_path, mask_future, free_fast=True):
    try:
        logger.info(f'start remove and set skin mask....')
        mask_clear, humanMask, control_image_return, normal_map_img, normal_3d_map_img, with_torso_mask, f_output_line_image_pil, seg_1 = mask_future.result()
        mask_np = np.array(mask_clear)
        logger.info(f'start LaMaModel remove closes')
        # model_instance = LaMaModel(model_path='/mnt/sessd/ai_tools/Inpaint-Anything/big-lama')
        clear_result, success_is = req_lama(file_path, mask_clear, free_fast)
        logger.info(f'success LaMaModel remove closes')
        if success_is:
            logger.info("start apply_skin_tone")
            image_bgr = cv2.imread(file_path)
            gen_extracted_textures = extract_texture(image_bgr, humanMask, mask_np)  # , min_YCrCb, max_YCrCb
            all_is_none = True
            for t_c_part, t_c_v in gen_extracted_textures.items():
                if t_c_v is not None:
                    all_is_none = False
                    break
            filled_image_pil = None
            if not all_is_none:
                extracted_textures = gen_extracted_textures
                filled_image = clear_result
                # 填充纹理
                for part, mask in humanMask.items():
                    texture = extracted_textures.get(part, None)
                    if texture is not None and np.count_nonzero(texture[:, :, 3]) > 0:
                        logger.info(f'{part} had wen li')
                        filled_image = warp_texture(filled_image, mask, texture, mask_np)
                    else:
                        logger.info(f'{part} had not wen li use before')
                        backup_texture, beforPart = find_non_empty_texture(extracted_textures)
                        if backup_texture is not None:
                            logger.info(f'{part} had not wen li use before and before is none use {beforPart}')
                            filled_image = warp_texture(filled_image, mask, backup_texture, mask_np)
                filled_image_pil = Image.fromarray(cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB))
            next_filled_image = clear_result
            # 填充纹理
                    # next_extracted_textures = get_def_extracted_textures()
                    # for part, mask in humanMask.items():
                    #     texture = next_extracted_textures.get(part, None)
                    #     next_filled_image = warp_texture(next_filled_image, mask, texture, mask_np)
                    # next_filled_image_pil = Image.fromarray(cv2.cvtColor(next_filled_image, cv2.COLOR_BGR2RGB))
            next_filled_image_pil = None
            clear_image_pil = Image.fromarray(cv2.cvtColor(clear_result, cv2.COLOR_BGR2RGB))
            # filled_image_pil.save(fix_gen_path)
            logger.info(f'finish remove and set skin mask....')
            return clear_image_pil, filled_image_pil, next_filled_image_pil
        else:
            logger.info("error LaMaModel")
    except Exception as ex:
        logger.info(ex)
        # 打印完整的异常堆栈
        traceback.print_exc()


def merge_masks(mask1, mask2):
    # 确保两个掩码的尺寸相同
    if mask1.shape != mask2.shape:
        raise ValueError("The shapes of mask1 and mask2 do not match.")

    # 进行按位或操作来合并掩码
    # merged_mask = np.maximum(mask1, mask2)
    # 合并掩码：如果任意一个掩码中有有效值（大于 0），则设置为有效（1）
    merged_ndarray = np.where((mask1 > 0) | (mask2 > 0), 255, 0)
    return merged_ndarray
def convert_to_ndarray(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError("Unsupported image type")
def free_re_auto_mask():
    url = "http://localhost:3060/"
    requests.get(url + 're')

def re_auto_mask(file_path, free_fast):
    url = "http://localhost:3060/"
    data = {'file_path': file_path}
    response = requests.post(url+'inpaint', data=data)
    # 检查响应状态
    if response.status_code == 200:
        try:
            # 使用 pickle 反序列化接收到的内容
            response_data = pickle.loads(response.content)
            densepose_mask_ndarray = response_data['densepose_mask_ndarray']
            close_mask = response_data['close_mask']
            not_close_mask_array = response_data['not_close_mask_array']
            segmentation_image_pil = response_data['segmentation_image_pil']
            # 处理每个数据
            logger.info(f'suc re_auto_mask api ')
            return densepose_mask_ndarray, close_mask, not_close_mask_array, segmentation_image_pil
        except Exception as e:
            logger.info(f"Error processing response: {e}")
            return None
        finally:
            if free_fast:
                free_re_auto_mask()
    else:
        logger.info(f"Request failed with status code {response.status_code}")
        return None

def re_auto_mask_with_out_head_b(file_path, filename, free_fast = True):
    logger.info(f"start pre handler file_path: {file_path} filename: {filename}")
    file_path = os.path.join(file_path, filename)
    image_pil = Image.open(file_path).convert("RGB")

    # 联合 人体姿势分析 将 身体手脚 头排除
    mask_future = main_executor.submit(gen_mask_f, file_path, image_pil, free_fast)

    # 删除衣服
    gen_fix_pic_future = main_executor.submit(gen_fix_pic, file_path, mask_future, free_fast)

    # 使用 ThreadPoolExecutor 并行处理每个部分
    fill_all_mask_future = None  # main_executor.submit(apply_skin_color, image_pil, remove_and_rege_mask)

    logger.info(f"finish submit trans pre handler")
    return mask_future, gen_fix_pic_future, fill_all_mask_future

def make_video_quicktime_compatible(input_path, output_path, target_width=1024, target_height=576):
    # 确保宽高为偶数
    target_width = target_width if target_width % 2 == 0 else target_width + 1
    target_height = target_height if target_height % 2 == 0 else target_height + 1
    """
    使用 ffmpeg 将视频重新编码为与 QuickTime 兼容的格式。

    :param input_path: 输入视频文件路径
    :param output_path: 输出视频文件路径
    :param target_width: 目标视频宽度
    :param target_height: 目标视频高度
    """
    command = [
        'ffmpeg',
        '-y',
        '-i', input_path,  # 输入文件路径
        '-vf', f'scale={target_width}:{target_height}',  # 设置缩放
        '-c:v', 'libx264',  # 使用 H.264 编码
        '-pix_fmt', 'yuv420p',  # 设置像素格式为 yuv420p
        '-movflags', '+faststart',  # 优化文件结构以便快速开始播放
        '-profile:v', 'baseline',  # 使用 H.264 baseline 配置文件
        '-level', '3.0',  # 设置 H.264 level
        '-an',  # 禁用音频流（如果没有音频）
        output_path  # 输出文件路径
    ]

    # 调用 ffmpeg 进行处理
    subprocess.run(command, check=True)

def i2v_processing(processed_data, app_path):
    url = "http://127.0.0.1:7860/generate-video/"
    filename = processed_data['filename']
    room_id = processed_data['roomId']
    i2v_image_pil_path = os.path.join(app_path, room_id, filename)
    target_width, target_height = Image.open(i2v_image_pil_path).size
    if not os.path.exists(i2v_image_pil_path):
        return
    # 要发送的图片文件
    files = {"file": open(i2v_image_pil_path, "rb")}
    # 参数
    data = {
        "seed": 42,
        "randomize_seed": False,
        "motion_bucket_id": 80,
        "fps_id": 8,
        "max_guidance_scale": 1.2,
        "min_guidance_scale": 1,
        "width": 1024,
        "height": 576,
        "num_inference_steps": 4,
        "decoding_t": 4
    }
    # 发送POST请求
    response = requests.post(url, files=files, data=data)
    i2v_video_name = filename + 'temp.mp4'
    return_i2v_video_name = filename + '.mp4'
    i2v_video_pil_path = os.path.join(app_path, room_id, i2v_video_name)
    return_i2v_video_pil_path = os.path.join(app_path, room_id, return_i2v_video_name)
    # 保存生成的视频
    with open(i2v_video_pil_path, "wb") as f:
        f.write(response.content)
    make_video_quicktime_compatible(i2v_video_pil_path, return_i2v_video_pil_path, target_width, target_height)
    logger.info("视频生成并保存成功！")
    return return_i2v_video_name
def get_hold_path(room_id, filename, app_path):
    return os.path.join(app_path, room_id, filename)

def get_fin_hold_path(room_id, filename):
    return os.path.join('/nvme0n1-disk/book_yes/static/uploads', room_id, filename)

def run_gen_it(processor, g_c_param: GenContextParam, notify_type, notify_fuc, app_path, room_image_manager):
    result_next,gen_it_prompt = processor.genIt(g_c_param)
    result_next = result_next.resize((g_c_param.original_width, g_c_param.original_height), Image.LANCZOS)
    processed_file_path = os.path.join(app_path, g_c_param.room_id, g_c_param.save_filename)
    result_next.save(processed_file_path)
    room_image_manager.insert_imgStr(g_c_param.room_id, f'{g_c_param.save_filename}', 'done',
                                     g_c_param.book_img_name, ext_info = gen_it_prompt,notify_fuc=notify_fuc,
                                     notify_type=notify_type)
    notify_fuc(notify_type, 'processing_step_fin', {'fin': 'f'}, to=g_c_param.room_id)
    del result_next

def room_path(room_id, app_path):
    return os.path.join(app_path, room_id)

def get_from_local(pil_img, save_img_path, conver = None):
    if pil_img:
        return pil_img
    elif isinstance(save_img_path, str) and os.path.exists(save_img_path):
        if conver:
            return Image.open(save_img_path).convert(conver)
        else:
            return Image.open(save_img_path)
    return None

def video_by_name(room_id, filename, port, org_faces_f, to_faces_f, notify_fuc, notify_type):
    file_face_name_video = f'p_vid{port}_{filename}'

    save_video_path = get_fin_hold_path(room_id, file_face_name_video)

    filename_replace_p = get_fin_hold_path(room_id, filename)
    replace_result_img = req_process_video(pic_b=filename_replace_p, pic_save=save_video_path,
                                           source_path_list=org_faces_f, target_path_list=to_faces_f, port=port)
    if replace_result_img is not None:

        # todo 增加发送视频
        notify_fuc(notify_type, 'processing_done',
                   {'img_type': 'done', 'video_url': save_video_path, 'filename': filename, 'text': '已完成'},
                   to=room_id,
                   keyList=get_rest_key())
    else:
        notify_fuc(notify_type, 'processing_step_progress',
                   {'text': '视频处理失败'}, to=room_id)
        notify_fuc(notify_type, 'processing_step_progress',
                   {'text': '可以继续上传需要替换的内容，或者切换模型使用其他功能'}, to=room_id,
                   keyList=get_rest_key())


def handle_image_inpaint(data, notify_fuc, app_path, room_image_manager, create_callback):
    # 处理并提取图片数据
    processed_data = process_req_data(data)
    notify_type = processed_data['notify_type']
    room_id = processed_data['roomId']
    # 从处理后的数据中提取值
    def_skin = processed_data['def_skin']
    filename = processed_data['filename']
    prompt = processed_data['prompt']
    prompt_2 = processed_data['prompt_2']
    re_p_float_array = processed_data['re_p_float_array']
    re_b_float_array = processed_data['re_b_float_array']
    ha_p = processed_data['ha_p']
    ga_b = processed_data['ga_b']
    reverse_prompt = processed_data['reverse_prompt']
    reverse_prompt_2 = processed_data['reverse_prompt_2']
    strength = processed_data['strength']
    num_inference_steps = processed_data['num_inference_steps']
    guidance_scale = processed_data['guidance_scale']
    seed = processed_data['seed']
    file_path = os.path.join(app_path, room_id, filename)
    # Generate a unique key for the input combination
    unique_key = generate_unique_key(room_id, filename, prompt, reverse_prompt, num_inference_steps, guidance_scale, seed, re_p_float_array, re_b_float_array, ha_p, ga_b, strength,prompt_2, reverse_prompt_2)
    logger.info(f"unique_key: {unique_key}")
    mask_future = None
    gen_fix_pic_future = None
    fill_all_mask_future = None
    filled_image_pil_return = None
    next_filled_image_pil_return = None
    clear_image_pil_return = None
    mask_clear_finally = None
    segmentation_image_pil = None
    control_image_return = None
    normal_map_img_return = None
    normal_3d_map_img_return = None
    with_torso_mask = None
    line_image_l_control = None
    fill_all_mask_img = None
    gen_save_path = GenSavePathParam(app_path, room_id, filename)
    if seed > -2:
        torch.manual_seed(seed)
    f_output_line_image_pil = None
    logger.info(f"filled_image_pil_filename: {gen_save_path.filled_image_pil_filename} ")
    notify_fuc(notify_type, 'processing_step_progress', {'text': 'book yes 识别图片...'},to=room_id)
    room_upload_folder = room_path(room_id, app_path)
    mask_future, gen_fix_pic_future, fill_all_mask_future = re_auto_mask_with_out_head_b(room_upload_folder, filename, True)

    if os.path.exists(gen_save_path.line_file_name_path):
        logger.info(f'open {gen_save_path.line_file_name_path} use it line page')
        line_image_l_control = Image.open(gen_save_path.line_file_name_path)
        logger.info(f'control_image is {type(line_image_l_control)}')

    image_org = Image.open(file_path).convert("RGB")
    original_width, original_height = image_org.size
    orgImage = image_org

    global sdxl_ac_in_cpu
    global sdxl_ac_in_gpu
    processor = None
    try:
        if mask_future is not None:
            mask_clear_finally, temp_humanMask, control_image_return, normal_map_img_return, normal_3d_map_img_return, with_torso_mask, f_output_line_image_pil, segmentation_image_pil = mask_future.result()
            room_image_manager.insert_imgStr(room_id, f'{gen_save_path.mask_clear_finally_filename}', 'pre_done', '重绘区域', 'k', mask_clear_finally, gen_save_path.mask_clear_finally_path, notify_fuc=notify_fuc, notify_type=notify_type)
            room_image_manager.insert_imgStr(room_id, f'{gen_save_path.control_net_fname}', 'pre_done', '姿势识别', 'k', control_image_return, gen_save_path.control_net_fname_path, notify_fuc=notify_fuc, notify_type=notify_type)
            if normal_map_img_return:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.nor_control_net_fname}', 'pre_done', '深度识别', 'k', normal_map_img_return, gen_save_path.nor_control_net_fname_path, notify_fuc=notify_fuc, notify_type=notify_type)
            if normal_3d_map_img_return:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.nor_3d_control_net_fname}', 'pre_done', '3d深度识别', 'k', normal_3d_map_img_return, gen_save_path.nor_3d_control_net_fname_path, notify_fuc=notify_fuc, notify_type=notify_type)
            if with_torso_mask:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.with_torso_mask_name}', 'pre_done', '最终重绘区域', 'k', with_torso_mask, gen_save_path.with_torso_mask_name_path, notify_fuc=notify_fuc, notify_type=notify_type)
            if f_output_line_image_pil:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.line_file_auto_name}', 'pre_done', '排除头部手臂的边线', 'k', f_output_line_image_pil, gen_save_path.line_file_auto_name_path, notify_fuc=notify_fuc, notify_type=notify_type)
            if segmentation_image_pil:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.mask_clear_finally_filename}', 'pre_done',
                                                 '分割图', 'k',
                                                 segmentation_image_pil, gen_save_path.segmentation_image_pil_path,
                                                 notify_fuc=notify_fuc, notify_type=notify_type)
            segmentation_image_pil = get_from_local(segmentation_image_pil, gen_save_path.segmentation_image_pil_path)

        notify_fuc(notify_type, 'processing_step_progress', {'text': 'book yes 加载模型中...'}, to=room_id)
        # processor =FluxInpaintPipelineSingleton ()
        logger.info('------------开始加载 vae unet ---------------')
        processor = sdxlInpainting(controlnet_list_t = re_p_float_array, use_out_test= (ga_b == 1.0))
        # processor = fluxInpainting(use_out_test=(ga_b == 1.0))
        if sdxl_ac_in_cpu:
            logger.info(f'--------------从 cpu加载模型到 gpu')
            torch.cuda.empty_cache()
            gc.collect()
            sdxl_ac_in_gpu = processor.pipe.to('cuda')
            logger.info(f'--------------完成 cpu加载模型到 gpu')
            sdxl_ac_in_cpu_del = sdxl_ac_in_cpu
            sdxl_ac_in_cpu = None
            del sdxl_ac_in_cpu_del
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f'--------------完成 释放 cpu空间-----------')
        else:
            sdxl_ac_in_gpu = processor.pipe
        notify_fuc(notify_type, 'processing_step_progress', {'text': 'book yes 计算中 共4张图可出稍等哈...'}, to=room_id)

        control_image = get_from_local(control_image_return, gen_save_path.control_net_fname_path)
        with_torso_mask_img = get_from_local(with_torso_mask, gen_save_path.with_torso_mask_name_path)
        auto_line_con = get_from_local(f_output_line_image_pil,gen_save_path.line_file_auto_name_path)

        if processor.use_out_test:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = processor.encode_and_cache(
                prompt, prompt_2, reverse_prompt, reverse_prompt_2)
            prompt_embeds = prompt_embeds.to('cuda')
            negative_prompt_embeds = negative_prompt_embeds.to('cuda')
            pooled_prompt_embeds = pooled_prompt_embeds.to('cuda')
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to('cuda')
        else:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = None,None,None,None

        # 不会执行
        if fill_all_mask_future is not None:
            fill_all_mask_img = fill_all_mask_future.result()
            if fill_all_mask_img is not None:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.fill_all_mask_img_fname}', 'pre_done', '使用默认颜色填充所有人体区域', 'k', fill_all_mask_img, gen_save_path.fill_all_mask_img_name_path, notify_fuc=notify_fuc, notify_type=notify_type)
        elif os.path.exists(gen_save_path.fill_all_mask_img_name_path):
            fill_all_mask_img = Image.open(gen_save_path.filled_image_pil_path)

        big_mask = get_from_local(mask_clear_finally, gen_save_path.mask_clear_finally_path, "L")
        normal_map_img = get_from_local(normal_map_img_return, gen_save_path.nor_control_net_fname_path)
        normal_map_3d_img = get_from_local(normal_3d_map_img_return, gen_save_path.nor_3d_control_net_fname_path)
        g_c_param = GenContextParam(prompt, prompt_2, reverse_prompt, reverse_prompt_2, orgImage, with_torso_mask_img,
                   num_inference_steps, guidance_scale,
                   create_callback('1/4', room_id, num_inference_steps, notify_type), strength,
                   [control_image, normal_map_img, segmentation_image_pil, line_image_l_control], re_b_float_array,
                   prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
                   original_width, original_height, '', room_id)
        g_c_param.next_genImage, g_c_param.save_filename = orgImage, f'p_1_{unique_key}.png'
        g_c_param.book_img_name = '原始生成'
        run_gen_it(processor, g_c_param, notify_type, notify_fuc, app_path, room_image_manager)
        # if fill_all_mask_img 可以执行 g_c_param.next_genImage = fill_all_mask_img g_c_param.save_filename = fill_all_processed_org_r_filename
        if gen_fix_pic_future is not None:
            clear_image_pil_return, filled_image_pil_return, next_filled_image_pil_return = gen_fix_pic_future.result()
            room_image_manager.insert_imgStr(room_id, f'{gen_save_path.cur_res_filename}', 'pre_done', '清除后图片', None, clear_image_pil_return, gen_save_path.cur_res_path, notify_fuc=notify_fuc, notify_type=notify_type)
            if filled_image_pil_return:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.filled_image_pil_filename}', 'pre_done', '自动识别填充' ,None,filled_image_pil_return, gen_save_path.filled_image_pil_path, notify_fuc=notify_fuc, notify_type=notify_type)
            if isinstance(next_filled_image_pil_return, Image.Image) and next_filled_image_pil_return is not None:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.next_filled_image_pil_filename}', 'pre_done', '默认填充', None, next_filled_image_pil_return, gen_save_path.next_filled_image_pil_path, notify_fuc=notify_fuc, notify_type=notify_type)

        clear_org_i = get_from_local(clear_image_pil_return, gen_save_path.cur_res_path)
        g_c_param.func_call_back, g_c_param.next_genImage, g_c_param.save_filename = create_callback('2/4', room_id, num_inference_steps, notify_type), clear_org_i,  f'p_2_{unique_key}.png'
        g_c_param.book_img_name = '擦出生成'
        run_gen_it(processor, g_c_param, notify_type, notify_fuc, app_path, room_image_manager)

        genImage = get_from_local(filled_image_pil_return, gen_save_path.filled_image_pil_path)

        if isinstance(genImage, Image.Image) and genImage is not None:
            g_c_param.func_call_back, g_c_param.next_genImage, g_c_param.save_filename = create_callback('3/4', room_id, num_inference_steps, notify_type), genImage,  f'p_3_{unique_key}.png'
            g_c_param.book_img_name = '填充纹理'
            run_gen_it(processor, g_c_param, notify_type, notify_fuc, app_path, room_image_manager)

        if auto_line_con:
            if normal_map_3d_img is not None:
                re_b_float_array[1], g_c_param.control_img_list[1] = re_b_float_array[2], None #normal_map_3d_img
            else:
                logger.info("did not set 3d nor------")
            g_c_param.control_img_list[3] = auto_line_con
            logger.info("use auto line con ------")
            g_c_param.func_call_back , g_c_param.next_genImage, g_c_param.save_filename = create_callback('4/4', room_id, num_inference_steps, notify_type), genImage,  f'p_4_{unique_key}.png'
            g_c_param.book_img_name = '预测遮挡'
            run_gen_it(processor, g_c_param, notify_type, notify_fuc, app_path, room_image_manager)
        next_genImage = get_from_local(next_filled_image_pil_return, gen_save_path.next_filled_image_pil_path)
        # if next_genImage 后 可以执行 默认填充 g_c_param.next_genImage = next_genImage 可以执行 g_c_param.save_filename = processed_r_r_filename

        notify_fuc(notify_type, 'processing_step_progress',
                  {'text': '可以继续上传需要替换的内容，或者切换模型使用其他功能'}, to=room_id,
                  keyList=get_rest_key())

    except Exception as e:
        logger.info(f"processing error is  -------: {e}")
        notify_fuc(notify_type, 'processing_step_progress', {'text': 'book yes 异常了请重新上传执行...'}, to=room_id)
    finally:
        logger.info(f'--------------推理完成开始 释放 gpu空间-----------')
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f'--------------释放 gpu空间 完成 开始移动 模型到 cpu-----------')
        sdxl_ac_in_cpu = processor.pipe.to("cpu")
        logger.info(f'--------------完成移动 模型到 cpu---开始释放GPU资源----------')
        sdxl_ac_in_gpu_del = sdxl_ac_in_gpu
        sdxl_ac_in_gpu = None
        del sdxl_ac_in_gpu_del
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f'--------------完成移动 模型到 cpu---完成释放GPU资源----------')

        # 释放 GPU 内存
        if 'prompt_embeds' in locals():
            del prompt_embeds
        if 'negative_prompt_embeds' in locals():
            del negative_prompt_embeds
        if 'pooled_prompt_embeds' in locals():
            del pooled_prompt_embeds
        if 'negative_pooled_prompt_embeds' in locals():
            del negative_pooled_prompt_embeds
        del big_mask
        torch.cuda.empty_cache()
        gc.collect()


def handle_image_processing_b(data, notify_fuc, app_path, room_image_manager, create_callback):
    # 处理并提取图片数据
    notify_type = data['notify_type']
    room_id = data['roomId']
    def_skin = data['def_skin']
    notify_fuc(notify_type, 'processing_step_progress', {'text': '到你了亲，开始处理...'}, to=room_id)

    if def_skin == '99':
        processed_data = process_req_data(data)
        i2v_video_name = i2v_processing(processed_data, app_path)
        room_image_manager.insert_imgStr(room_id, f'{i2v_video_name}', 'done', 'book', notify_fuc=notify_fuc, notify_type=notify_type)
        notify_fuc(notify_type, 'processing_step_fin', {'fin': 'f'}, to=room_id)
        return
    # 情景
    if def_skin == '909':
        prompt = data['prompt']
        filename = data['filename']
        face_filename = data['face_filename']
        gen_type = data['gen_type']
        file_path = os.path.join(app_path, room_id, filename)
        unique_key = generate_unique_key(room_id, filename, prompt, face_filename, gen_type)
        logger.info(f"unique_key: {unique_key}")
        if face_filename is not None and face_filename != '':
            face_filename = get_hold_path(room_id, face_filename, app_path)
        logger.info(f'chose face_filename is {face_filename}')
        en_prompt= translate_baidu(prompt)

        text_gen_img_s = req_text_gen(file_path, en_prompt, only_face_path=face_filename, gen_type=gen_type)
        for idx, text_gen_img in enumerate(text_gen_img_s):
            file_txt_name = f'p_txt_{idx}_{unique_key}.png'
            logger.info(f"Image {idx} saved to {file_txt_name}")
            file_txt_name_path = os.path.join(app_path, room_id, file_txt_name)
            room_image_manager.insert_imgStr(room_id, f'{file_txt_name}', 'done', '生成图', file_i=text_gen_img,
                                             file_p=file_txt_name_path, ext_info=en_prompt, notify_fuc=notify_fuc,
                                             notify_type=notify_type)

        # 1006 flux ip
        text_gen_img_s_t = req_text_gen(file_path, en_prompt, only_face_path=face_filename, gen_type=gen_type, port=1006)
        for idx, text_gen_img in enumerate(text_gen_img_s_t):
            file_txt_name = f'p_flux_{idx}_{unique_key}.png'
            logger.info(f"Image {idx} saved to {file_txt_name}")
            file_txt_name_path = os.path.join(app_path, room_id, file_txt_name)
            room_image_manager.insert_imgStr(room_id, f'{file_txt_name}', 'done','生成图', file_i = text_gen_img,
                                         file_p = file_txt_name_path, ext_info=en_prompt, notify_fuc=notify_fuc,
                                         notify_type=notify_type)

        notify_fuc(notify_type, 'processing_step_progress',
                   {'text': '已完成，可以继续上传需要替换的内容，或者切换模型使用其他功能'}, to=room_id, keyList=get_rest_key())
        return
    # filename
    if def_skin == 'face':
        filename = data['filename']
        file_path = os.path.join(app_path, room_id, filename)
        face_images = req_get_face(file_path)
        for idx, face_gen_img in enumerate(face_images):
            new_unique_id = str(uuid.uuid4())
            file_face_name = f'p_f_{idx}_{new_unique_id}.png'
            logger.info(f"Image {idx} saved to {file_face_name}")
            file_face_name_path = os.path.join(app_path, room_id, file_face_name)
            # 创建 InlineKeyboardMarkup
            keyboard_face = [
                [
                    InlineKeyboardButton("使用此主角", callback_data=f"o_f_i_{file_face_name}")
                ]
            ]
            reply_face_markup = InlineKeyboardMarkup(keyboard_face)
            room_image_manager.insert_imgStr(room_id, f'{file_face_name}', 'done','生成图', file_i = face_gen_img,
                                         file_p = file_face_name_path, ext_info='huanlian', notify_fuc=notify_fuc,
                                         notify_type=notify_type,keyList=reply_face_markup)
        return
    # filename org_faces to_faces
    if def_skin == 'start_swap_face':
        filename = data['filename']
        org_faces = data['org_faces']
        to_faces = data['to_faces']
        org_faces_f = []
        for s in org_faces:
            org_faces_f.append(get_fin_hold_path(room_id, s))
        to_faces_f = []
        for s in to_faces:
            to_faces_f.append(get_fin_hold_path(room_id, s))

        filename_replace_p = get_fin_hold_path(room_id, filename)
        replace_result_img_s = []
        replace_result_img_5000 = req_replace_face(pic_b=filename_replace_p, source_path_list=org_faces_f, target_path_list=to_faces_f, port=5000)
        replace_result_img_s.append(replace_result_img_5000)
        replace_result_img_1005 = req_replace_face(pic_b=filename_replace_p, source_path_list=org_faces_f,
                                                   target_path_list=to_faces_f, port=1005)
        replace_result_img_s.append(replace_result_img_1005)

        for i, replace_result_img in enumerate(replace_result_img_s):
            if replace_result_img is not None:
                file_face_name = f'p_ref{i}_{filename}'
                file_face_name_path = os.path.join(app_path, room_id, file_face_name)
                room_image_manager.insert_imgStr(room_id, f'{file_face_name}', 'done', '换脸结果图', file_i=replace_result_img,
                                                 file_p=file_face_name_path, ext_info='huanlian', notify_fuc=notify_fuc,
                                                 notify_type=notify_type)
            else:
                notify_fuc(notify_type, 'processing_step_progress',
                          {'text': '图像中为识别到您需要替换的特征图片'}, to=room_id)
                notify_fuc(notify_type, 'processing_step_progress',
                          {'text': '可以继续上传需要替换的内容，或者切换模型使用其他功能'}, to=room_id,
                          keyList=get_rest_key())
        notify_fuc(notify_type, 'processing_step_progress',
                   {'text': '已完成，可以继续上传需要替换的内容，或者切换模型使用其他功能'}, to=room_id,
                   keyList=get_rest_key())
        return
    # filename org_faces to_faces
    if def_skin == 'start_swap_face_video':
        filename = data['filename']
        org_faces = data['org_faces']
        to_faces = data['to_faces']

        org_faces_f = []
        for s in org_faces:
            org_faces_f.append(get_fin_hold_path(room_id, s))
        to_faces_f = []
        for s in to_faces:
            to_faces_f.append(get_fin_hold_path(room_id, s))
        video_by_name(room_id, filename, 1005, org_faces_f, to_faces_f, notify_fuc, notify_type)
        video_by_name(room_id, filename, 5000, org_faces_f, to_faces_f, notify_fuc, notify_type)
        return
    # pre_face_pic_list
    if def_skin == 'swap_face':
        pre_face_pic_list = data['pre_face_pic_list']
        total_faces = []
        for per_sinle_face_pic in pre_face_pic_list:
            face_pat = get_hold_path(room_id, per_sinle_face_pic, app_path)
            face_images = req_get_face(face_pat)
            total_faces.extend(face_images)
        to_keyboard_face = []
        media_group = []
        new_unique_id = str(uuid.uuid4())
        for idx, face_gen_img in enumerate(total_faces):
            file_face_name = f'mf_{idx}_{new_unique_id}.png'
            to_keyboard_face.append([
                InlineKeyboardButton(f"选择 {file_face_name}", callback_data=f"mu_chose_{file_face_name}")
            ])
            logger.info(f"Image {idx} saved to {file_face_name}")
            file_face_name_path = os.path.join(app_path, room_id, file_face_name)
            room_image_manager.insert_imgStr(room_id, f'{file_face_name}', 'face_pre','识别脸图', file_i = face_gen_img,
                                         file_p = file_face_name_path, ext_info='', notify_fuc=notify_fuc,
                                         notify_type='none')
            media_group.append(InputMediaPhoto(open(file_face_name_path, 'rb'), caption=f"图片{file_face_name}"))
        to_keyboard_face.append([
            InlineKeyboardButton(f"重新选择", callback_data=f"reset_pre_face")
        ])
        reply_markup_face = InlineKeyboardMarkup(to_keyboard_face)
        notify_fuc(notify_type, 'processing_done',
                   {'img_type': 'media_group', 'text':'逐个选择配对，先选A在选B则为将A替换为B，你可以先后选择多组'}, to=room_id,
                   keyList=reply_markup_face, media_group=media_group)
        return
    if def_skin == 'inpaint':
        # 重绘任务
        handle_image_inpaint(data, notify_fuc, app_path, room_image_manager, create_callback)
    return


