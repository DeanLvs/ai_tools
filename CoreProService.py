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
import zipfile

main_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

main_executor_control = concurrent.futures.ThreadPoolExecutor(max_workers=16)

class GenContextParam:
    def __init__(self, prompt, prompt_2, reverse_prompt, reverse_prompt_2,
                        next_genImage, big_mask, num_inference_steps, guidance_scale,
                        func_call_back, strength, control_img_list,
                        control_float_array, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
                        negative_pooled_prompt_embeds, original_width, original_height, processed_filename, room_id):
        self.prompt = prompt
        self.prompt_2 = prompt_2
        self.reverse_prompt = reverse_prompt
        self.reverse_prompt_2 = reverse_prompt_2
        self.next_genImage = next_genImage
        self.big_mask = big_mask
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.func_call_back = func_call_back
        self.strength = strength
        self.control_img_list = control_img_list
        self.control_float_array = control_float_array
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        self.original_width = original_width
        self.original_height = original_height
        self.save_filename = processed_filename
        self.room_id = room_id
        self.book_img_name = 'book'

class GenSavePathParam:
    def __init__(self, base_path, uq_id, filename):
        self.filled_image_pil_filename = 'filled_image_pil_' + filename
        self.filled_image_pil_path = os.path.join(base_path, uq_id, self.filled_image_pil_filename)

        self.next_filled_image_pil_filename = 'filled_image_pil_next_' + filename
        self.next_filled_image_pil_path = os.path.join(base_path, uq_id, self.next_filled_image_pil_filename)

        self.cur_res_filename = 'clear_return_' + filename
        self.cur_res_path = os.path.join(base_path, uq_id, self.cur_res_filename)

        self.mask_clear_finally_filename = 'filled_use_mask_image_pil_' + filename
        self.mask_clear_finally_path = os.path.join(base_path, uq_id, self.mask_clear_finally_filename)

        self.segmentation_image_pil_filename = 'seg_use_image_pil_' + filename
        self.segmentation_image_pil_path = os.path.join(base_path, uq_id,
                                                   self.segmentation_image_pil_filename)

        self.control_net_fname = f'control_net_' + filename
        self.control_net_fname_path = os.path.join(base_path, uq_id, self.control_net_fname)

        self.nor_control_net_fname = f'nor_control_net_' + filename
        self.nor_control_net_fname_path = os.path.join(base_path, uq_id, self.nor_control_net_fname)

        self.nor_3d_control_net_fname = f'nor_3d_control_net_' + filename
        self.nor_3d_control_net_fname_path = os.path.join(base_path, uq_id, self.nor_3d_control_net_fname)

        self.with_torso_mask_name = f'with_torso_mask_' + filename
        self.with_torso_mask_name_path = os.path.join(base_path, uq_id, self.with_torso_mask_name)

        self.line_file_name = f"line_{filename}"
        self.line_file_name_path = os.path.join(base_path, uq_id, self.line_file_name)

        self.line_file_auto_name = f"line_auto_{filename}"
        print(f'save find and get {self.line_file_auto_name}')
        self.line_file_auto_name_path = os.path.join(base_path, uq_id, self.line_file_auto_name)

        self.fill_all_mask_img_fname = f'fill_all_skin_' + filename
        self.fill_all_mask_img_name_path = os.path.join(base_path, uq_id, self.fill_all_mask_img_fname)

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

    # 发起请求
    response = requests.post(url, data=data)

    # 检查响应状态
    if response.status_code == 200:
        zip_data = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_data, 'r') as zf:
            # 读取 d_pose_resized 和 masks
            d_pose_resized = zf.read("d_pose_resized.png")
            d_3d_canny_resized = zf.read("d_3d_canny_resized.png")
            masks = zf.read("masks.pkl")
            return d_pose_resized, pickle.loads(masks), d_3d_canny_resized
    else:
        print(f"Request failed with status code {response.status_code}")
        return None, None

def gen_mask_f(file_path_e, file_i_e, mask_i_e, not_close_mask_img_e):
    def process_human_mask(file_path, file_i, mask_i, not_close_mask_img):
        d_pose_resized, humanMask, d_3d_canny_resized = dense_req(file_path)
        print(f'完成 check pose....')
        exclude_mask = humanMask['exclude']
        all_body_mask = humanMask['all_body']
        # 删除键
        if 'exclude' in humanMask:
            del humanMask['exclude']
        if 'all_body' in humanMask:
            del humanMask['all_body']
        print(f'get d_3d_canny_resize')
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
        return clear_mask, humanMask, with_torso_mask, output_line_image_pil, d_pose_resized

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
    def generate_normal_map(image_pil, next_image_pil):
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
        response = requests.post("http://localhost:5070/inpaint", timeout=99999, files=files)
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
                    return depth_org_img, depth_3d_img
            except Exception as e:
                print(f"depth Error processing response: {e}")
                return None, None
        else:
            print(f"depth Request failed with status code {response.status_code}")
            return None, None

    process_human_mask_f = main_executor_control.submit(process_human_mask, file_path_e, file_i_e, mask_i_e, not_close_mask_img_e)
    detect_control_image_f = main_executor_control.submit(detect_control_image, file_i_e)
    clear_mask, humanMask, with_torso_mask, f_output_line_image_pil, d_pose_resized_r = process_human_mask_f.result()
    control_image = detect_control_image_f.result()
    # 使用3d图出深度图
    generate_normal_map_f = main_executor_control.submit(generate_normal_map, d_pose_resized_r, file_i_e)
    normal_map_img, normal_3d_map_img  = generate_normal_map_f.result()
    return clear_mask, humanMask, control_image, normal_map_img, normal_3d_map_img, with_torso_mask, f_output_line_image_pil

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
        mask_clear, humanMask, control_image_return, normal_map_img, normal_3d_map_img, with_torso_mask, f_output_line_image_pil = mask_future.result()
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

    # 删除衣服
    gen_fix_pic_future = main_executor.submit(gen_fix_pic, file_path, mask_future)

    print(f"finish trans wenli an clear")
    return mask_future, gen_fix_pic_future, segmentation_image_pil
