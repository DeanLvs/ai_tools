import os
import torch
import numpy as np
from downloadC import download_file_c
import hashlib
import time
import traceback
import scipy.ndimage
from PIL import ImageDraw,Image, ImageFilter
from ImageProcessorSDX import CustomInpaintPipeline
from mask_generator import MaskGenerator  # 导入 MaskGenerator 类
import gc
from concurrent.futures import ThreadPoolExecutor
import cv2
from sklearn.cluster import KMeans
def cluster_colors(image, k=3):
    # 形状转换为 (num_pixels, 3) 的二维数组
    pixels = image.reshape(-1, 3)

    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    new_colors = kmeans.cluster_centers_[kmeans.labels_]

    # 将聚类结果转换回原始形状
    clustered_image = new_colors.reshape(image.shape).astype(np.uint8)
    return clustered_image

def smooth_color_distribution(image, blur_radius=5):
    # 对区域进行均值平滑
    smoothed_image = cv2.blur(image, (blur_radius, blur_radius))
    return smoothed_image
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
def gen_fix_pic(file_path, file_i, mask_i, not_close_mask_img, extracted_textures = None):
    from lama_handler import LaMaModel
    import cv2
    try:
        print(f'begin check pose....')
        from SupMaskSu import handlerMaskAll
        humanMask = handlerMaskAll(file_path)
        print(f'finish check pose....')
        exclude_mask = humanMask['exclude']
        # all_body_mask = humanMask['all_body']
        # 删除键
        if 'exclude' in humanMask:
            del humanMask['exclude']
        if 'all_body' in humanMask:
            del humanMask['all_body']
        print(f'it ..... {exclude_mask}')
        clear_mask = remove_exclude_mask(file_i, mask_i, exclude_mask)
        clear_mask = remove_exclude_mask(file_i, clear_mask, not_close_mask_img)
        mask_clear = clear_mask
        mask_np = np.array(mask_clear)
        print(f'start LaMaModel remove closes')
        model_instance = LaMaModel(model_path='/mnt/sessd/ai_tools/Inpaint-Anything/big-lama')
        clear_result, success_is = model_instance.predict(file_path, None, mask_clear)
        print(f'success LaMaModel remove closes')
        # fix_f_gen_path = os.path.join(app.config['UPLOAD_FOLDER'], 'clear_f_' + file_name)
        # cv2.imwrite(fix_f_gen_path, cur_res)
        if success_is:
            print("start apply_skin_tone")
            from SupMaskSu import apply_skin_tone, extract_texture, warp_texture, \
                find_non_empty_texture
            image_bgr = cv2.imread(file_path)
            if not extracted_textures:
                extracted_textures = extract_texture(image_bgr, humanMask, mask_np)  # , min_YCrCb, max_YCrCb
                # 重新初始化 filled_image
                filled_image = clear_result
            else:
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
            # 保存结果
            filled_image_pil = Image.fromarray(cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB))
            print(f'finish check pose an ge wenli....')
            return clear_result, filled_image_pil, mask_clear
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
    close_mask,not_close_mask = mask_generator.generate_mask(image_pil)
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
    return densepose_mask_ndarray,close_mask, convert_to_ndarray(not_close_mask)

def re_auto_mask_with_out_head_b(image_org, file_path, extracted_textures = None):
    remove_and_rege_mask, close_mask_img, not_close_mask_img = re_auto_mask(file_path)
    merged_mask_pixels = np.argwhere(remove_and_rege_mask > 0)
    merged_mask_pixels = [[int(x), int(y)] for y, x in merged_mask_pixels]
    mask_clear = Image.new("L", image_org.size, 0)
    draw = ImageDraw.Draw(mask_clear)
    for point in merged_mask_pixels:
        draw.point(point, fill=255)
    print(f"finish draw close mask")
    clear_result, filled_image_pil, mask_clear_finally= gen_fix_pic(file_path, image_org, mask_clear, not_close_mask_img, extracted_textures)
    print(f"finish trans wenli an clear")
    return mask_clear, clear_result, filled_image_pil, mask_clear_finally


def process_set_lora(data):
    print("got it process_set_lora")
    room_id = data['roomId']
    res_text = '无参数请输出 c 的 l 或者 w'
    if 'lora_id' in data and data['lora_id'] != '':
        lora_id = data['lora_id']
        if lora_id.isdigit():
            print(f'load this {lora_id}')
            response, output_path, filename, total_size = download_file_c(lora_id)
            CHUNK_SIZE = 1638400
            output_file = os.path.join(output_path, filename)
            if not os.path.exists(output_file):
                with open(output_file, 'wb') as f:
                    downloaded = 0
                    start_time = time.time()
                    print(f'start do this {start_time}')
                    while True:
                        buffer = response.read(CHUNK_SIZE)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        f.write(buffer)
            else:
                print(f'had down load this {lora_id} {output_file}')
                output_file = os.path.join(output_path, filename)
        else:
            output_file = f"/mnt/sessd/civitai-downloader/lora/{lora_id}"
        print(f'load this {lora_id} {output_file}')
        processor = CustomInpaintPipeline()
        processor.unload_lora_weights()
        processor.load_lora_weights(output_file)
        res_text = '完成' + output_file
    elif 'wei_id' in data and data['wei_id'] != '':
        wei_id = data['wei_id']
        processor = CustomInpaintPipeline()
        weights_path = f"/mnt/sessd/civitai-downloader/{wei_id}"
        processor.load_pretrained_weights(weights_path)


def get_def_extracted_textures():
    skin_texture = np.array([
        [[255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255],
         [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255]],
        [[255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255],
         [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255]],
        [[255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255],
         [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255]],
        [[255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255],
         [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255]],
        [[255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255],
         [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255]],
        [[255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255],
         [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255]],
        [[255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255],
         [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255]],
        [[255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255],
         [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255]],
        [[255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255],
         [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255], [255, 240, 230, 255]]
    ], dtype=np.uint8)
    texture = cv2.cvtColor(skin_texture, cv2.COLOR_BGRA2RGBA)

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
    return parts_to_include
def genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, genImage, big_mask,
          num_inference_steps=50, guidance_scale=8, progress_callback=None, strength=0.951):
    try:
        print(f"gen use prompt: {prompt} reverse_prompt: {reverse_prompt} "
              f"num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}, "
              f"strength: {strength} prompt_2: {prompt_2} reverse_prompt_2: {reverse_prompt_2}")
        with torch.no_grad():
            resultTemp = processor.pipe(
                prompt = prompt,
                prompt_2 = prompt_2,
                negative_prompt= reverse_prompt,
                negative_prompt_2 = reverse_prompt_2,
                image=genImage,
                mask_image=big_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback=progress_callback,
                strength=strength,
                callback_steps=1  # 调用回调的步数
            ).images[0]
    finally:
        # 释放 GPU 内存
        torch.cuda.empty_cache()
        gc.collect()
    return resultTemp

def handle_image_processing_b(processor=None, file_path="/mnt/sessd/civitai-downloader/trans", file_name = "", model_name=""):
    print("got it process_image_b")
    img_path = f'/mnt/sessd/civitai-downloader/test/{file_name}'
    image_org = Image.open(img_path)
    original_width, original_height = image_org.size
    image = image_org.convert("RGB")
    torch.manual_seed(78)
    clear_img_path = f'{file_path}/m_c_{file_name}'
    gen_skin_img_path = f'{file_path}/m_g_{file_name}'
    mask_img_path =  f'{file_path}/m_mask_{file_name}'
    if not os.path.exists(clear_img_path) or not os.path.exists(gen_skin_img_path) or not os.path.exists(mask_img_path):
        mask_clear_return, clear_return, filled_image_pil_return, gen_img_mask_finally = re_auto_mask_with_out_head_b(image, img_path)
        del mask_clear_return
        cv2.imwrite(clear_img_path, clear_return)
        filled_image_pil_return.save(gen_skin_img_path)
        gen_img_mask_finally.save(mask_img_path)
    else:
        clear_return = Image.open(clear_img_path).convert("RGB")
        filled_image_pil_return = Image.open(gen_skin_img_path).convert("RGB")
        gen_img_mask_finally = Image.open(mask_img_path).convert("L")
    model_name = model_name.replace(' ', '')
    org_img_path_result = genIt(processor, "remove clothes, nude",
                   "big asses, big boobs",
                   "deformed, bad anatomy, mutated, long neck,disconnected limbs",
                   "unnaturally contorted position, unnaturally thin waist", image, gen_img_mask_finally)
    org_img_path_result = org_img_path_result.resize((original_width, original_height), Image.LANCZOS)
    org_img_path_result.save(f'{file_path}/org_{model_name}{file_name}')

    clear_img_result = genIt(processor, "remove clothes, nude",
                   "big asses, big boobs",
                   "deformed, bad anatomy, mutated, long neck,disconnected limbs",
                   "unnaturally contorted position, unnaturally thin waist", clear_return,
                   gen_img_mask_finally)
    clear_img_result = clear_img_result.resize((original_width, original_height), Image.LANCZOS)
    clear_img_result.save(f'{file_path}/clear_{model_name}{file_name}')
    gen_skin_result = genIt(processor, "remove clothes, nude",
                   "big asses, big boobs",
                   "deformed, bad anatomy, mutated, long neck,disconnected limbs",
                   "unnaturally contorted position, unnaturally thin waist", filled_image_pil_return,
                   gen_img_mask_finally)
    gen_skin_result = gen_skin_result.resize((original_width, original_height), Image.LANCZOS)
    gen_skin_result.save(f'{file_path}/gen_{model_name}{file_name}')
    del image

    del clear_return
    del filled_image_pil_return
    del gen_img_mask_finally
    del org_img_path_result
    del clear_img_result
    del gen_skin_result
    torch.cuda.empty_cache()
    gc.collect()
if __name__ == '__main__':
    processor = CustomInpaintPipeline()
    model_name_list = ['epicrealismXL_v8KISS-inpainting.safetensors','haveallsdxlInpaint_v10.safetensors','inpaintSDXLPony_inpaintPony.safetensors','inpaintSDXLPony_juggerInpaintV8.safetensors','juggernautXL_v9rdphoto2Inpaint.safetensors','juggernautXL_versionXInpaint.safetensors','lustifySDXLNSFW_v10-inpainting.safetensors','realvisxlv40_-inpainting.safetensors','realvisxlV40_v30InpaintBakedvae.safetensors','realismFromHadesXL_v60XLInpainting.safetensors','sevenof9PonyRealMix_inpaintPony.safetensors','imaginariumInpaint_v10.safetensors','talmendoxlSDXL_v11Beta.safetensors','jibMixRealisticXL_v140crystalclarity.safetensors']
    directory_path = '/mnt/sessd/civitai-downloader/test/'
    for model_name in model_name_list:
        weights_path = f"/mnt/sessd/civitai-downloader/{model_name}"
        processor.load_pretrained_weights(weights_path)
        print(f'run {model_name}')
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            print(f'run {model_name} {filename}')
            handle_image_processing_b(processor = processor, model_name=model_name, file_name=filename)


