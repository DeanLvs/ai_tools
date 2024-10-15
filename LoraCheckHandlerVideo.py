import os
import torch
import argparse
import numpy as np
import traceback

from PIL import ImageDraw,Image
from ImageProcessorSDX import CustomInpaintPipeline
import gc
import cv2

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
def gen_fix_pic(file_path, file_i, mask_i, not_close_mask_img, fix_gen_path, file_name = "clear"):
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

            # 初始状态为 True，假设所有纹理都为 None
            all_none = True
            min_YCrCb = np.array([87, 138, 107])
            max_YCrCb = np.array([255, 146, 128])
            max_iterations = 15  # 最大循环次数
            iteration_count = 0  # 当前循环次数

            while all_none and iteration_count < max_iterations:
                # 提取纹理
                extracted_textures = extract_texture(image_bgr, humanMask, mask_np, min_YCrCb, max_YCrCb)
                # 重新初始化 filled_image
                filled_image = clear_result

                # 检查提取的纹理是否全部为 None
                for part, mask in humanMask.items():
                    texture = extracted_textures.get(part, None)
                    if texture is not None:
                        all_none = False  # 如果找到至少一个非 None 的纹理，设置为 False
                        break  # 跳出 for 循环

                # 调整 min_YCrCb 和 max_YCrCb 的值
                if min_YCrCb[0] > 0:
                    min_YCrCb[0] = max(min_YCrCb[0] - 50, 0)  # 确保不会减到负值
                elif max_YCrCb[1] < 255:
                    max_YCrCb[1] = min(max_YCrCb[1] + 50, 255)
                elif min_YCrCb[1] > 0:
                    min_YCrCb[1] = max(min_YCrCb[1] - 50, 0)
                elif max_YCrCb[2] < 255:
                    max_YCrCb[2] = min(max_YCrCb[2] + 50, 255)
                elif min_YCrCb[2] > 0:
                    min_YCrCb[2] = max(min_YCrCb[2] - 50, 0)
                iteration_count += 1  # 增加循环

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
    from mask_generator import MaskGenerator  # 导入 MaskGenerator 类
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
    print(f"finish check close mask: {file_path}")
    return densepose_mask_ndarray,close_mask, convert_to_ndarray(not_close_mask)

def re_auto_mask_with_out_head_b(file_path,image_pil):
    remove_and_rege_mask, close_mask_img, not_close_mask_img = re_auto_mask(file_path)
    merged_mask_pixels = np.argwhere(remove_and_rege_mask > 0)
    merged_mask_pixels = [[int(x), int(y)] for y, x in merged_mask_pixels]
    mask_clear = Image.new("L", image_pil.size, 0)
    draw = ImageDraw.Draw(mask_clear)
    for point in merged_mask_pixels:
        draw.point(point, fill=255)
    print(f"finish draw close mask")
    clear_result, filled_image_pil, mask_clear_finally = gen_fix_pic(file_path, image_pil, mask_clear,
                                                                     not_close_mask_img, None, None)
    print(f"finish trans wenli an clear")
    return mask_clear, clear_result, filled_image_pil, mask_clear_finally

def genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, genImage, big_mask,
          num_inference_steps, guidance_scale, progress_callback, strength=0.8):
    try:
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

def handle_image_processing_b(filePath, prompt="",
                              reverse_prompt="",
                              prompt_2=None,
                              reverse_prompt_2= None,
                              num_inference_steps=80, guidance_scale=9,seed=None,strength=0.9, mask_clear_finally=None, filled_image_pil_return=None, lora_path=None):
    print(f'prompt {prompt} lora_path {lora_path}')
    image_pil = Image.open(filePath).convert("RGB")
    a, b = image_pil.size
    processor = CustomInpaintPipeline()
    processor.load_lora_weights(f"/mnt/sessd/civitai-downloader/lora/{lora_path}")
    try:
        if seed:
            torch.manual_seed(seed)
        org_fix_Result = genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, filled_image_pil_return,
                               mask_clear_finally, num_inference_steps, guidance_scale, None, strength)
        org_fix_Result = org_fix_Result.resize((a, b), Image.LANCZOS)
        return org_fix_Result
    finally:
        torch.cuda.empty_cache()
        gc.collect()
#

def get_upper_half_mask(image):
    # Open the input image
    width, height = image.size

    # Create a new image with the same size as the input image
    mask = Image.new('L', (width, height), 0)

    # Create a draw object to draw the mask
    draw = ImageDraw.Draw(mask)

    # Define the upper two-thirds rectangle (covering the upper two-thirds of the image)
    upper_two_thirds = [(0, 0), (width, height * 2 // 3)]

    # Draw the upper two-thirds rectangle in white (255)
    draw.rectangle(upper_two_thirds, fill=255)

    # Save the mask image
    return mask
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process an image and apply a mask.")
    parser.add_argument("l", type=str, help="Path to the input lora")
    parser.add_argument("p", type=str, help="Pprompt")

    args = parser.parse_args()
    filled_image_pil_return = Image.open("/mnt/sessd/ai_tools/00030-2169183031.png").convert("RGB")
    mask_clear_finally = get_upper_half_mask(filled_image_pil_return)
    img = handle_image_processing_b("/mnt/sessd/ai_tools/00030-2169183031.png", mask_clear_finally=mask_clear_finally,
                                    filled_image_pil_return=filled_image_pil_return, lora_path = args.l,prompt=args.p)
    img.save("loratest.png")