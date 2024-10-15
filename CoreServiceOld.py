import os
import numpy as np
import hashlib
import traceback
import scipy.ndimage
from PIL import ImageDraw,Image, ImageFilter
from concurrent.futures import ThreadPoolExecutor
import cv2
from mask_generator import MaskGenerator  # 导入 MaskGenerator 类
from SupMaskSu import handlerMaskAll
from lama_handler import LaMaModel
import concurrent.futures

main_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

main_executor_control = concurrent.futures.ThreadPoolExecutor(max_workers=16)

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


def resize_image(image_path, max_size=2048): #
    image = Image.open(image_path)
    width, height = image.size

    if width > max_size or height > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        image.save(image_path)

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


def remove_exclude_mask(image_pil, mask_clear, exclude_mask):
    # 将 mask_clear 转换为 NumPy 数组
    mask_clear_ndarray = np.array(mask_clear)

    # 确保 exclude_mask 的大小与 mask_clear_ndarray 一致
    if mask_clear_ndarray.shape != exclude_mask.shape:
        raise ValueError("The shapes of mask_clear and exclude_mask do not match.")

    # 从 mask_clear_ndarray 中去除 exclude_mask 的区域
    result_ndarray = np.where(exclude_mask > 0, 0, mask_clear_ndarray)

    # 将结果转换回 PIL 图像
    final_mask = Image.fromarray(result_ndarray.astype(np.uint8))

    return final_mask

def gen_mask_f(file_path_e, file_i_e, mask_i_e, not_close_mask_img_e):
    def process_human_mask(file_path, file_i, mask_i, not_close_mask_img):
        humanMask = handlerMaskAll(file_path)
        print(f'finish check pose....')
        exclude_mask = humanMask['exclude']
        all_body_mask = humanMask['all_body']
        # 删除键
        if 'exclude' in humanMask:
            del humanMask['exclude']
        if 'all_body' in humanMask:
            del humanMask['all_body']

        # # 创建一个彩色图像，与掩码图像大小相同，初始值全为 0
        # color_image = np.zeros((all_body_mask.shape[0], all_body_mask.shape[1], 3), dtype=np.uint8)
        # # 定义浅蓝色和浅黄色的 RGB 颜色
        # light_blue = [255, 165, 0]  # 浅蓝色 (R, G, B)
        # light_yellow = [255, 235, 205]  # 浅黄色 (R, G, B)
        # # 将掩码中的黑色 (0) 替换为浅蓝色
        # color_image[all_body_mask == 0] = light_blue

        # 将掩码中的白色 (1) 替换为浅黄色
        # color_image[all_body_mask == 255] = light_yellow

        # 将结果转换回 PIL 图像
        # color_image_r = Image.fromarray(np.array(color_image))

        color_image_r = Image.fromarray(all_body_mask.astype(np.uint8))

        print(f'it ..... {exclude_mask}')
        clear_mask = remove_exclude_mask(file_i, mask_i, exclude_mask)
        clear_mask = remove_exclude_mask(file_i, clear_mask, not_close_mask_img)
        return clear_mask, humanMask, color_image_r

    process_human_mask_f = main_executor_control.submit(process_human_mask, file_path_e, file_i_e, mask_i_e, not_close_mask_img_e)
    # detect_control_image_f = main_executor_control.submit(detect_control_image, file_i_e)
    # generate_normal_map_f = main_executor_control.submit(generate_normal_map, file_i_e)

    clear_mask, humanMask, color_image_r = process_human_mask_f.result()
    # control_image = detect_control_image_f.result()
    # normal_map_img = generate_normal_map_f.result()
    return clear_mask, humanMask, control_image, normal_map_img, color_image_r

def gen_clear_pic(file_path, mask_path):
    try:
        print(f'begin clear img....')
        model_instance = LaMaModel(model_path='/mnt/sessd/ai_tools/Inpaint-Anything/big-lama')
        clear_result, success_is = model_instance.predict(file_path, mask_path, None)
        print(f'success LaMaModel remove closes')
        if success_is:
            print("start apply_skin_tone")
            clear_image_pil = Image.fromarray(cv2.cvtColor(clear_result, cv2.COLOR_BGR2RGB))
            return clear_image_pil
        else:
            print("error LaMaModel")
            return None
    except Exception as ex:
        print(ex)
        # 打印完整的异常堆栈
        traceback.print_exc()

def apply_skin_color(image_pil, mask):
    if True:
        return None
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

def gen_fix_pic(file_path, mask_future):
    try:
        print(f'begin check pose....')
        mask_clear, humanMask, control_image_return, normal_map_img, corler = mask_future.result()
        mask_np = np.array(mask_clear)
        print(f'start LaMaModel remove closes')
        model_instance = LaMaModel(model_path='/mnt/sessd/ai_tools/Inpaint-Anything/big-lama')
        clear_result, success_is = model_instance.predict(file_path, None, mask_clear)
        print(f'success LaMaModel remove closes')
        if success_is:
            print("start apply_skin_tone")
            from SupMaskSu import apply_skin_tone, extract_texture, warp_texture, \
                find_non_empty_texture
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
                        print(f'{part} had wen li')
                        filled_image = warp_texture(filled_image, mask, texture, mask_np)
                    else:
                        print(f'{part} had not wen li use before')
                        backup_texture, beforPart = find_non_empty_texture(extracted_textures)
                        if backup_texture is not None:
                            print(f'{part} had not wen li use before and before is none use {beforPart}')
                            filled_image = warp_texture(filled_image, mask, backup_texture, mask_np)
                filled_image_pil = Image.fromarray(cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB))
            next_filled_image = clear_result
            # 填充纹理
            next_extracted_textures = get_def_extracted_textures()
            for part, mask in humanMask.items():
                texture = next_extracted_textures.get(part, None)
                next_filled_image = warp_texture(next_filled_image, mask, texture, mask_np)
            next_filled_image_pil = Image.fromarray(cv2.cvtColor(next_filled_image, cv2.COLOR_BGR2RGB))

            clear_image_pil = Image.fromarray(cv2.cvtColor(clear_result, cv2.COLOR_BGR2RGB))
            # filled_image_pil.save(fix_gen_path)
            print(f'finish check pose an ge wenli....')
            return clear_image_pil, filled_image_pil, next_filled_image_pil
        else:
            print("error LaMaModel")
    except Exception as ex:
        print(ex)
        # 打印完整的异常堆栈
        traceback.print_exc()


def merge_masks(mask1, mask2):
    mask1 = convert_to_ndarray(mask1)
    mask2 = convert_to_ndarray(mask2)
    # 确保两个掩码的尺寸相同
    if mask1.shape != mask2.shape:
        raise ValueError("The shapes of mask1 and mask2 do not match.")

    # 进行按位或操作来合并掩码
    merged_mask = np.maximum(mask1, mask2)

    return merged_mask
def convert_to_ndarray(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError("Unsupported image type")
def re_auto_mask(file_path):
    print(f"begin check close mask: {file_path}")
    # 生成两个掩码
    mask_generator = MaskGenerator()
    image_pil = Image.open(file_path).convert("RGB")
    close_mask,not_close_mask,segmentation_image_pil = mask_generator.generate_mask_and_seg(image_pil)
    # 转换掩码为 ndarrays
    print(f'begin close mask convert_to_ndarray')
    densepose_mask_ndarray = convert_to_ndarray(close_mask)
    print(f'finish close mask convert_to_ndarray')
    # densepose_mask_ndarray = dilate_mask_np(densepose_mask_ndarray, iterations=8)
    print(f'begin dilate_mask_by_percentage close mask')
    densepose_mask_ndarray = dilate_mask_by_percentage(densepose_mask_ndarray, 3)
    print(f'finish dilate_mask_by_percentage close mask')
    # human_mask_ndarray = convert_to_ndarray(humanMask)
    # 合并掩码
    # merged_mask = merge_masks(densepose_mask_ndarray, human_mask_ndarray)
    # 合并相掉
    # merged_mask = subtract_masks(densepose_mask_ndarray, human_mask_ndarray)
    print(f"finish check close mask: {file_path}")
    return densepose_mask_ndarray, close_mask, convert_to_ndarray(not_close_mask), segmentation_image_pil

def re_auto_mask_with_out_head_b(file_path, filename):
    print(f"  file_path: {file_path} filename: {filename}")
    file_path = os.path.join(file_path, filename)
    image_pil = Image.open(file_path).convert("RGB")
    remove_and_rege_mask, close_mask_img, not_close_mask_img, segmentation_image_pil = re_auto_mask(file_path)
    merged_mask_pixels = np.argwhere(remove_and_rege_mask > 0)
    merged_mask_pixels = [[int(x), int(y)] for y, x in merged_mask_pixels]
    mask_clear = Image.new("L", image_pil.size, 0)
    draw = ImageDraw.Draw(mask_clear)
    for point in merged_mask_pixels:
        draw.point(point, fill=255)
    print(f"finish draw close mask")

    # 使用 ThreadPoolExecutor 并行处理每个部分
    mask_future = main_executor.submit(gen_mask_f, file_path, image_pil, mask_clear, not_close_mask_img)

    # 使用 ThreadPoolExecutor 并行处理每个部分
    fill_all_mask_future = main_executor.submit(apply_skin_color, image_pil, remove_and_rege_mask)

    gen_fix_pic_future = main_executor.submit(gen_fix_pic, file_path, mask_future)

    print(f"finish trans wenli an clear")
    return mask_future, gen_fix_pic_future, fill_all_mask_future, segmentation_image_pil
