import os
import numpy as np
import hashlib
import traceback
import scipy.ndimage
from PIL import ImageDraw,Image, ImageFilter
from concurrent.futures import ThreadPoolExecutor
import cv2
from SupMaskSu import extract_texture, warp_texture, find_non_empty_texture
import concurrent.futures
import requests, io
import pickle

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


def resize_image(image_path, max_size=2048):
    # 打开图片
    image = Image.open(image_path)
    width, height = image.size

    # 获取文件的扩展名，并进行大小写敏感检查
    file_root, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()  # 转换为小写

    # 如果扩展名是 .jpg 或 .jpeg，将其转换为小写 .jpg
    if file_extension in ['.jpg', '.jpeg']:
        new_image_path = file_root + ".jpg"
    # 如果扩展名是 .png，将其转换为小写 .png
    elif file_extension == '.png':
        new_image_path = file_root + ".png"
    else:
        # 如果不是 jpg 或 png，转换为 png
        new_image_path = file_root + ".png"
        image = image.convert("RGB")  # 确保图片转为RGB模式以保存为 png

    # 如果图片尺寸大于 max_size，则调整尺寸
    if width > max_size or height > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)

    # 保存修改后的图片
    image.save(new_image_path)
    print(f"Image saved at {new_image_path}")
    return os.path.basename(new_image_path)

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


def dense_req(file_path):
    url = "http://localhost:9080/inpaint"
    data = {'file_path': file_path}
    response = requests.post(url, data=data)
    # 检查响应状态
    if response.status_code == 200:
        try:
            masks = pickle.loads(response.content)
            return masks
        except Exception as e:
            print(f"dense_req Error processing response: {e}")
            return None
    else:
        print(f"dense_req Request failed with status code {response.status_code}")
        return None

def gen_mask_f(file_path_e, file_i_e, mask_i_e, not_close_mask_img_e):
    def process_human_mask(file_path, file_i, mask_i, not_close_mask_img):
        humanMask = dense_req(file_path)
        print(f'完成 check pose....')
        exclude_mask = humanMask['exclude']
        all_body_mask = humanMask['all_body']
        # 删除键
        if 'exclude' in humanMask:
            del humanMask['exclude']
        if 'all_body' in humanMask:
            del humanMask['all_body']
        output_line_image_pil = None
        if 'Torso_line' in humanMask:
            torso_line = humanMask['Torso_line']
            left_line = humanMask['left_line']
            right_line = humanMask['right_line']
            np.save('torso_line.npy', torso_line)
            np.save('left_line.npy', left_line)
            np.save('right_line.npy', right_line)
            # 创建一个空的输出图像，大小与 torso_line 相同
            output_line_image = np.zeros_like(torso_line)
            del humanMask['Torso_line']
            del humanMask['left_line']
            del humanMask['right_line']
            # 对 torso_line 使用 Canny 边缘检测，提取身体外部边缘
            edges_torso = cv2.Canny(torso_line.astype(np.uint8), 100, 200)

            # 对 left_line 和 right_line 分别使用 Canny 边缘检测
            edges_left = cv2.Canny(left_line.astype(np.uint8), 100, 200)
            edges_right = cv2.Canny(right_line.astype(np.uint8), 100, 200)

            # 计算腿部与身体的距离，找到相邻区域
            distance_left = cv2.distanceTransform(1 - left_line.astype(np.uint8), cv2.DIST_L2, 3)
            distance_right = cv2.distanceTransform(1 - right_line.astype(np.uint8), cv2.DIST_L2, 3)

            # 阈值设定，改为3像素的距离作为相邻判断
            left_adjacent = (distance_left <= 3) & (torso_line > 0)
            right_adjacent = (distance_right <= 3) & (torso_line > 0)

            # 移除相邻区域的边界
            edges_torso[left_adjacent] = 0
            edges_left[left_adjacent] = 0
            edges_right[right_adjacent] = 0

            # 将处理后的 torso 边缘叠加到输出图像上
            output_line_image = np.maximum(output_line_image, edges_torso)

            # 将处理后的左腿和右腿的边缘叠加到输出图像上
            output_line_image = np.maximum(output_line_image, edges_left)
            output_line_image = np.maximum(output_line_image, edges_right)
            # 将结果转换为 PIL 图像
            output_line_image_pil = Image.fromarray(output_line_image)

        print(f'it ..... {exclude_mask}')
        # 剔除掉 手 脚
        # 将 mask_clear 转换为 NumPy 数组
        mask_i_ndarray = np.array(mask_i)
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
        return clear_mask, humanMask, with_torso_mask, output_line_image_pil

    # 姿势图
    def detect_control_image(image_pil):
        # 获取图像的原始格式
        img_format = image_pil.format if image_pil.format else 'PNG'
        # 将 PIL.Image 对象转换为字节流
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format=img_format)  # 使用原始图像格式
        img_byte_arr.seek(0)  # 将流的指针移动到开头
        # 以二进制形式发送图像数据
        files = {'file_i': img_byte_arr}
        # 发起 POST 请求
        response = requests.post("http://localhost:5060/inpaint", timeout=99999, files=files)
        # 检查响应状态
        if response.status_code == 200:
            try:
                control_image = pickle.loads(response.content)
                # 返回结果图像
                return control_image
            except Exception as e:
                print(f"open pose Error processing response: {e}")
                return None
        else:
            print(f"open pose Request failed with status code {response.status_code}")
            return None
    # 阴影图
    def generate_normal_map(image_pil):
        # 获取图像的原始格式
        img_format = image_pil.format if image_pil.format else 'PNG'
        # 将 PIL.Image 对象转换为字节流
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format=img_format)  # 使用原始图像格式
        img_byte_arr.seek(0)  # 将流的指针移动到开头
        # 以二进制形式发送图像数据
        files = {'file_i': img_byte_arr}
        # 发起 POST 请求
        response = requests.post("http://localhost:5070/inpaint", timeout=99999, files=files)
        # 检查响应状态
        if response.status_code == 200:
            try:
                # 将字节流反序列化
                control_image = pickle.loads(response.content)
                # 返回结果图像
                return control_image
            except Exception as e:
                print(f"depth Error processing response: {e}")
                return None
        else:
            print(f"depth Request failed with status code {response.status_code}")
            return None

    process_human_mask_f = main_executor_control.submit(process_human_mask, file_path_e, file_i_e, mask_i_e, not_close_mask_img_e)
    detect_control_image_f = main_executor_control.submit(detect_control_image, file_i_e)
    generate_normal_map_f = main_executor_control.submit(generate_normal_map, file_i_e)

    clear_mask, humanMask, with_torso_mask, f_output_line_image_pil = process_human_mask_f.result()
    control_image = detect_control_image_f.result()
    normal_map_img = generate_normal_map_f.result()
    #
    return clear_mask, humanMask, control_image, normal_map_img, with_torso_mask, f_output_line_image_pil

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

def req_lama(file_path, mask_clear):
    # 将 PIL.Image 对象转换为字节流
    img_io = io.BytesIO()
    mask_clear.save(img_io, format='PNG')  # 你可以选择其他格式
    img_io.seek(0)  # 将文件指针重置到开始位置

    # 发起 HTTP POST 请求
    url = "http://localhost:9090/inpaint"
    files = {'mask_clear': ('mask.png', img_io, 'image/png')}
    data = {
        'file_path': file_path
    }
    response = requests.post(url, files=files, data=data)

    # 处理响应的图像
    if response.status_code == 200:
        img_io_re = io.BytesIO(response.content)
        img_pil = Image.open(img_io_re)
        print(f'-----suc return lam---------')
        return np.array(img_pil), True
    print(f'-----error return lam---------')
    return None, False

def gen_fix_pic(file_path, mask_future):
    try:
        print(f'begin check pose....')
        mask_clear, humanMask, control_image_return, normal_map_img, with_torso_mask, f_output_line_image_pil = mask_future.result()
        mask_np = np.array(mask_clear)
        print(f'start LaMaModel remove closes')
        # model_instance = LaMaModel(model_path='/mnt/sessd/ai_tools/Inpaint-Anything/big-lama')
        clear_result, success_is = req_lama(file_path, mask_clear)
        print(f'success LaMaModel remove closes')
        if success_is:
            print("start apply_skin_tone")
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
                    # next_extracted_textures = get_def_extracted_textures()
                    # for part, mask in humanMask.items():
                    #     texture = next_extracted_textures.get(part, None)
                    #     next_filled_image = warp_texture(next_filled_image, mask, texture, mask_np)
                    # next_filled_image_pil = Image.fromarray(cv2.cvtColor(next_filled_image, cv2.COLOR_BGR2RGB))
            next_filled_image_pil = None
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
def re_auto_mask(file_path):
    url = "http://localhost:3060/inpaint"
    data = {'file_path': file_path}
    response = requests.post(url, data=data)
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
            print(f'suc re_auto_mask api ')
            return densepose_mask_ndarray, close_mask, not_close_mask_array, segmentation_image_pil
        except Exception as e:
            print(f"Error processing response: {e}")
            return None
    else:
        print(f"Request failed with status code {response.status_code}")
        return None

def re_auto_mask_with_out_head_b(file_path, filename):
    print(f"  file_path: {file_path} filename: {filename}")
    file_path = os.path.join(file_path, filename)
    image_pil = Image.open(file_path).convert("RGB")
    # remove_and_rege_mask 放大后的衣服区域
    remove_and_rege_mask, close_mask_img, not_close_mask_img, segmentation_image_pil = re_auto_mask(file_path)
    merged_mask_pixels = np.argwhere(remove_and_rege_mask > 0)
    merged_mask_pixels = [[int(x), int(y)] for y, x in merged_mask_pixels]
    mask_clear = Image.new("L", image_pil.size, 0)
    draw = ImageDraw.Draw(mask_clear)
    for point in merged_mask_pixels:
        draw.point(point, fill=255)
    print(f"finish draw close mask")

    # 联合 人体姿势分析 将 身体手脚 头排除
    mask_future = main_executor.submit(gen_mask_f, file_path, image_pil, mask_clear, not_close_mask_img)

    # 使用 ThreadPoolExecutor 并行处理每个部分
    fill_all_mask_future = main_executor.submit(apply_skin_color, image_pil, remove_and_rege_mask)

    # 删除衣服
    gen_fix_pic_future = main_executor.submit(gen_fix_pic, file_path, mask_future)

    print(f"finish trans wenli an clear")
    return mask_future, gen_fix_pic_future, fill_all_mask_future, segmentation_image_pil
