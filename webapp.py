import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import numpy as np
import hashlib
import scipy.ndimage
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from PIL import Image, ImageDraw, ImageFilter
from ImageProcessor import ImageProcessor
from ImageProcessorSDX import CustomInpaintPipeline
from mask_generator import MaskGenerator  # 导入 MaskGenerator 类
from SupMaskSu import handlerMask
from concurrent.futures import ThreadPoolExecutor, as_completed
from lama_handler import LaMaModel
import cv2
import gc

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
socketio = SocketIO(app)
# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Dictionary to store processed results
MAX_SIZE = 2048  # 最大宽度和高度
processed_results = {}

@app.route('/')
def index():
    return render_template('index.html')

def generate_unique_key(*args):
    # Concatenate all arguments to create a unique string
    unique_string = '_'.join(map(str, args))
    # Create a hash of the unique string
    return hashlib.md5(unique_string.encode()).hexdigest()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'filename': filename, 'file_url': f'/uploads/{filename}'})

def dilate_mask(mask, iterations):
    mask_array = np.array(mask)
    dilated_mask = scipy.ndimage.binary_dilation(mask_array, iterations=iterations)
    dilated_mask = (dilated_mask * 255).astype(np.uint8)
    return Image.fromarray(dilated_mask)


def dilate_mask_outward_only(mask, iterations):
    # 将掩码转换为 numpy 数组
    mask_array = np.array(mask)
    # 进行膨胀操作
    dilated_mask = scipy.ndimage.binary_dilation(mask_array, iterations=iterations)
    # 进行腐蚀操作
    eroded_mask = scipy.ndimage.binary_erosion(mask_array, iterations=iterations)
    # 计算形态学梯度（只保留外部膨胀部分）
    outward_only_dilated_mask = dilated_mask & ~eroded_mask
    # 转换回 uint8 类型，并乘以 255 以恢复到二值图像格式
    outward_only_dilated_mask = (outward_only_dilated_mask * 255).astype(np.uint8)

    # 将 numpy 数组转换为 PIL 图像并返回
    return Image.fromarray(outward_only_dilated_mask)

@app.route('/generate_mask', methods=['POST'])
def generate_mask():
    data = request.json
    filename = data['filename']
    points = data['points']
    points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]  # 将点转换为坐标对
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Load image and create mask
    image = Image.open(file_path).convert("RGB")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for point in points:
        draw.point(point, fill=255)
    # 对掩码进行模糊处理，使边界更平滑
    # mask = mask.filter(ImageFilter.GaussianBlur(5))

    replace_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
    mask.save(replace_mask_path)

    # Save big mask
    big_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'big_mask_' + filename)
    mask.save(big_mask_path)
    return jsonify({'mask_url': f'/uploads/big_mask_' + filename})


def mask_to_pixel_positions(mask):
    # 使用 np.nonzero 获取掩码中非零值的坐标
    y_indices, x_indices = np.nonzero(mask)
    # 将 y_indices 和 x_indices 转换成坐标对列表
    pixel_positions = list(zip(x_indices, y_indices))
    return pixel_positions

@app.route('/auto_recognize', methods=['POST'])
def auto_recognize():
    data = request.json
    filename = data['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # 调用 re_auto_mask 获取合并后的掩码
    merged_mask = re_auto_mask(filename)

    # 获取合并掩码的像素位置
    merged_mask_pixels = np.argwhere(merged_mask > 0)

    # 转换为原生 Python int 类型
    merged_mask_pixels = [[int(x), int(y)] for y, x in merged_mask_pixels]

    return jsonify({'mask_pixels': merged_mask_pixels})

def subtract_masks(mask1, mask2):
    mask1 = convert_to_ndarray(mask1)
    mask2 = convert_to_ndarray(mask2)
    # 确保两个掩码的尺寸相同
    if mask1.shape != mask2.shape:
        raise ValueError("The shapes of mask1 and mask2 do not match.")

    # 进行按位操作来计算 mask1 中的非 mask2 部分
    subtracted_mask = np.maximum(mask1 - mask2, 0)

    return subtracted_mask

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
def re_auto_mask(filename, re_p=0, re_b=0, ha_p=0, ga_b=0):
    print(f"  re_p: {re_p} re_p: {re_p} or ha_p: {ha_p}  or ga_b: {ga_b} ")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Load image

    # 生成两个掩码
    mask_generator = MaskGenerator()
    image_pil = Image.open(file_path).convert("RGB")
    close_mask = mask_generator.generate_mask(image_pil)
    # humanMask = handlerMask(img)
    # 转换掩码为 ndarrays
    # human_mask_ndarray = convert_to_ndarray(humanMask)
    # densepose_mask_ndarray = convert_to_ndarray(close_mask)
    # 合并掩码
    # merged_mask = merge_masks(human_mask_ndarray, densepose_mask_ndarray)
    # mask = merged_mask
    mask = close_mask
    # Create big mask by dilating the original mask
    replace_mask = dilate_mask(close_mask, iterations=re_p)  # 调整iterations以控制扩展范围
    if re_b > 0:
        replace_mask = replace_mask.filter(ImageFilter.GaussianBlur(re_b))
    # Save big mask
    replace_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
    replace_mask.save(replace_mask_path)
    # Create big mask by dilating the original mask
    big_mask = dilate_mask(mask, iterations=ha_p)  # 调整iterations以控制扩展范围
    if ga_b > 0:
        big_mask = big_mask.filter(ImageFilter.GaussianBlur(ga_b))
    # Save big mask
    big_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'big_mask_' + filename)
    big_mask.save(big_mask_path)
    return mask


from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFilter
import os

def re_auto_mask_with_out_head(filename, re_p=0, re_b=0, ha_p=0, ga_b=0,rm_p = 10):
    print(f"  re_p: {re_p} re_b: {re_b} or ha_p: {ha_p}  or ga_b: {ga_b} ")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    mask_generator = MaskGenerator()
    image_pil = Image.open(file_path).convert("RGB")
    #需要处理的掩码位置
    mask = mask_generator.generate_mask(image_pil)

    # 生成需要替换的掩码，可以向外扩散部分像素，优化便捷
    def process_replace_mask():
        # Create big mask by dilating the original mask
        if re_p > 0:
            replace_mask = dilate_mask(mask, iterations=re_p)  # 调整iterations以控制扩展范围
        else:
            replace_mask = mask
        if re_b > 0:
            replace_mask = replace_mask.filter(ImageFilter.GaussianBlur(re_b))
        # Save big mask
        replace_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
        replace_mask.save(replace_mask_path)
        # 转换掩码为 ndarrays
        return replace_mask_path

    # 生成需要重绘的掩码，向外扩散大一些，让模型更好绘制
    def process_big_mask():
        # Create big mask by dilating the original mask
        if ha_p > 0:
            big_mask = dilate_mask(mask, iterations=ha_p)  # 调整iterations以控制扩展范围
        else:
            big_mask = mask
        if ga_b > 0:
            big_mask = big_mask.filter(ImageFilter.GaussianBlur(ga_b))
        # Save big mask
        big_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'big_mask_' + filename)
        big_mask.save(big_mask_path)
        return big_mask_path

    # 生成需要移除的掩码，控制下大小保证边缘
    def process_remove_mask():
        big_fix_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fix_mask_' + filename)
        if rm_p > 0:
            kernel = np.ones((3, 3), np.uint8)
            count_mask = convert_to_ndarray(mask)
            fix_mask_dilated = cv2.dilate(count_mask, kernel, iterations=rm_p)
            # 确保 fix_mask 是一个值在 0 和 255 之间的 uint8 类型数组
            fix_mask_dilated = np.clip(fix_mask_dilated, 0, 255).astype(np.uint8)
            # 将 fix_mask_dilated 转换为单通道图像并保存
            fix_mask_image = Image.fromarray(fix_mask_dilated, mode='L')
            fix_mask_image.save(big_fix_path)
        else:
            mask.save(big_fix_path)
        fix_processed_filename = 'fix_processed_' + filename
        fix_processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], fix_processed_filename)
        try:
            model_instance = LaMaModel(model_path='/mnt/sessd/ai_tools/Inpaint-Anything/big-lama')
            # cur_res, success_is = model_instance.predict(file_path, fix_mask_path)
            cur_res, success_is = model_instance.predict(file_path, big_fix_path)
            if success_is:
                cv2.imwrite(fix_processed_file_path, cur_res)
        except Exception as ex:
            print(ex)

        return big_fix_path

    with ThreadPoolExecutor() as executor:
        future_replace_mask_f = executor.submit(process_replace_mask)
        future_big_mask_f = executor.submit(process_big_mask)
        future_remove_mask_f =  executor.submit(process_remove_mask)
        replace_mask_f = future_replace_mask_f.result()
        big_mask_f = future_big_mask_f.result()
        remove_mask_f = future_remove_mask_f.result()
    return mask


def re_auto_mask_with_out_head_b(filename, re_p=10, re_b=0, ha_p=10, ga_b=0,rm_p = 10):
    print(f"  re_p: {re_p} re_b: {re_b} or ha_p: {ha_p}  or ga_b: {ga_b} ")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mask_generator = MaskGenerator()
    image_pil = Image.open(file_path).convert("RGB")
    #需要处理的掩码位置
    mask = mask_generator.generate_mask(image_pil)

    # 生成需要替换的掩码，可以向外扩散部分像素，优化便捷
    def process_replace_mask():
        # Create big mask by dilating the original mask
        if re_p > 0:
            replace_mask = dilate_mask(mask, iterations=re_p)  # 调整iterations以控制扩展范围
        else:
            replace_mask = mask
        if re_b > 0:
            replace_mask = replace_mask.filter(ImageFilter.GaussianBlur(re_b))
        # Save big mask
        replace_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
        replace_mask.save(replace_mask_path)
        # 转换掩码为 ndarrays
        return replace_mask_path

    # 生成需要重绘的掩码，向外扩散大一些，让模型更好绘制
    def process_big_mask():
        # Create big mask by dilating the original mask
        if ha_p > 0:
            big_mask = dilate_mask(mask, iterations=ha_p)  # 调整iterations以控制扩展范围
        else:
            big_mask = mask
        if ga_b > 0:
            big_mask = big_mask.filter(ImageFilter.GaussianBlur(ga_b))
        # Save big mask
        big_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'big_mask_' + filename)
        big_mask.save(big_mask_path)
        return big_mask_path

    with ThreadPoolExecutor() as executor:
        future_replace_mask_f = executor.submit(process_replace_mask)
        future_big_mask_f = executor.submit(process_big_mask)
        replace_mask_f = future_replace_mask_f.result()
        big_mask_f = future_big_mask_f.result()

    return big_mask_f


@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['roomId']
    session_id = request.sid
    join_room(room_id)
    print(f'User {session_id} joined room {room_id}')


@socketio.on('join')
def handle_join(data):
    room_id = data['roomId']
    # 将客户端的socket.id加入到指定的房间
    socketio.enter_room(data['sid'], room_id)
    print(f'User {data["sid"]} joined room {room_id}')
# 连接时自动加入一个以 session ID 为房间名的房间
@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    join_room(session_id)

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    leave_room(session_id)

@socketio.on('process_image')
def handle_image_processing(data):
    filename = data['filename']
    prompt = data['prompt']
    room_id = data['roomId']
    re_mask = data['re_mask']
    re_p = data['re_p']
    re_b = int(data['re_b'])
    ha_p =  data['ha_p']
    ga_b = data['ga_b']
    if not prompt or prompt == "":
        prompt = "R/bigasses,R/pussy,R/boobs,nude,Really detailed skin,reddit,Match the original pose,Match the original image angle,Match the light of the original image,natural body proportions" #consistent angles, original pose,seamless background integration, matching lighting and shading, matching skin tone, no deformation
    reverse_prompt = data['reverse_prompt']
    if not reverse_prompt or reverse_prompt == "":
        reverse_prompt = "deformed,bad anatomy, mutated,long neck,disconnected limbs" #underweight body,
    num_inference_steps = data['num_inference_steps']
    if not num_inference_steps:
        num_inference_steps = 120
    else:
        num_inference_steps = int(num_inference_steps)
    guidance_scale = data['guidance_scale']
    if not guidance_scale:
        guidance_scale = 8.7
    else:
        guidance_scale = float(guidance_scale)
    seed = data['seed']
    # 设置随机种子
    if not seed:
        seed = 56
    else:
        seed = int(seed)
    torch.manual_seed(seed)
    print(f"prompt: {prompt} reverse_prompt: {reverse_prompt} "
          f"num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}, "
          f"seed: {seed}")
    mask_filename = 'mask_' + filename
    big_mask_filename = 'big_mask_' + filename

    # Generate a unique key for the input combination
    unique_key = generate_unique_key(filename, prompt, reverse_prompt, num_inference_steps, guidance_scale, seed, re_p, re_b, ha_p, ga_b)
    print(f"unique_key: {unique_key} processed_results: {processed_results} ")
    if unique_key in processed_results:
        processed_filename = processed_results[unique_key]
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_filename}'}, broadcast=True, to=room_id)
        return

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
    big_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], big_mask_filename)

    had_suc = False
    if re_mask != 'F' or not os.path.exists(mask_path) or not os.path.exists(big_mask_path):
        print(f"no mask_path: {mask_path} or big_mask_path: {big_mask_path} ")
        re_auto_mask_with_out_head(filename, int(re_p), int(re_b), int(ha_p), int(ga_b))

    fix_processed_filename = 'fix_processed_' + filename
    fix_processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], fix_processed_filename)

    if os.path.exists(fix_processed_file_path):
        emit('processing_done', {'processed_image_url': f'/uploads/{fix_processed_filename}'}, broadcast=True,
             to=room_id)
        had_suc = True

    processor = ImageProcessor(checkpoint_paths=[
        "/mnt/sessd/civitai-downloader/pornmasterPro_v8-inpainting-b.safetensors"
        # "realisticVisionV60B1_v51HyperInpaintVAE.safetensors"
        # ,"cleftofvenusV1.safetensors"
        ])

    # Generate image with progress updates
    mask = Image.open(mask_path).convert("L")
    big_mask = Image.open(big_mask_path).convert("L")
    # Save processed image
    processed_filename = f'processed_{unique_key}.png'
    f_f_processed_filename = f'f_f_processed_{unique_key}.png'
    processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    f_f_processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f_f_processed_filename)
    emit('processing_done_key', {'processing_done_key': f'{processed_filename}'}, broadcast=True, to=room_id)

    def progress_callback(step, t, latents):
        progress = int((step / num_inference_steps) * 100)
        emit('processing_progress', {'progress': progress}, broadcast=True, to=room_id)
    image_org = Image.open(file_path)
    original_width, original_height = image_org.size
    image = image_org.convert("RGB")
    # model_instance = LaMaModel(model_path='/mnt/sessd/ai_tools/Inpaint-Anything/big-lama')
    try:
        with torch.no_grad():
            result = processor.pipe(
                prompt = prompt,
                reverse_prompt= reverse_prompt,
                image=image,
                mask_image=big_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback=progress_callback,
                callback_steps=1  # 调用回调的步数
            ).images[0]
        # 调整生成图像尺寸为原始图像的尺寸
        # processed_r_filename = f'processed_r_or_{unique_key}.png'
        # processed_r_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_r_filename)
        # result.save(processed_r_file_path)
        # emit('processing_done', {'processed_image_url': f'/uploads/{processed_r_filename}'}, broadcast=True, to=room_id)
        result = result.resize((original_width, original_height), Image.LANCZOS)
        processed_r_filename = f'processed_r_{unique_key}.png'
        processed_r_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_r_filename)
        # 将 PIL 图像转换为 NumPy 数组
        result_np = np.array(result)

        # 如果是 RGB 图像，转换为 BGR 以符合 OpenCV 的格式
        result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(processed_r_file_path, result_np)
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_r_filename}'}, broadcast=True, to=room_id)
        # 对掩码进行高斯模糊处理
        blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=8))

        blended_result = Image.composite(result, image, blurred_mask)
        # blended_result = model_instance.inpaint_and_blend(result, image, blurred_mask)
        blended_result.save(processed_file_path)
        processed_results[unique_key] = processed_filename
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_filename}'}, broadcast=True, to=room_id)

        if had_suc:
            image_org_fix = Image.open(fix_processed_file_path)
            image_fix = image_org_fix.convert("RGB")
            blended_result_fixed = Image.composite(result, image_fix, blurred_mask)
            # blended_result_fixed = model_instance.inpaint_and_blend(result, image_fix, blurred_mask)
            blended_result_fixed.save(f_f_processed_file_path)
            processed_results[unique_key] = processed_filename
            emit('processing_done', {'processed_image_url': f'/uploads/{f_f_processed_filename}'}, broadcast=True,
                 to=room_id)
    finally:
        # 释放 GPU 内存
        del result
        del image
        del big_mask
        torch.cuda.empty_cache()
@socketio.on('process_image_b')
def handle_image_processing_b(data):
    print("got it")
    filename = data['filename']
    prompt = data['prompt']
    room_id = data['roomId']
    re_mask = data['re_mask']
    re_p = data['re_p']
    re_b = int(data['re_b'])
    ha_p =  data['ha_p']
    ga_b = data['ga_b']
    if not prompt or prompt == "":
        prompt = "R/bigasses,R/pussy,R/boobs,nude,Really detailed skin,reddit,Match the original pose,Match the original image angle,Match the light of the original image,natural body proportions" #consistent angles, original pose,seamless background integration, matching lighting and shading, matching skin tone, no deformation
    reverse_prompt = data['reverse_prompt']
    if not reverse_prompt or reverse_prompt == "":
        reverse_prompt = "deformed,bad anatomy, mutated,long neck,disconnected limbs" #underweight body,
    num_inference_steps = data['num_inference_steps']
    if not num_inference_steps:
        num_inference_steps = 120
    else:
        num_inference_steps = int(num_inference_steps)
    guidance_scale = data['guidance_scale']
    if not guidance_scale:
        guidance_scale = 8.7
    else:
        guidance_scale = float(guidance_scale)
    seed = data['seed']
    # 设置随机种子
    if not seed:
        seed = 56
    else:
        seed = int(seed)
    torch.manual_seed(seed)
    print(f"prompt: {prompt} reverse_prompt: {reverse_prompt} "
          f"num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}, "
          f"seed: {seed}")

    big_mask_filename = 'big_mask_' + filename
    replace_filename = 'mask_' + filename
    # Generate a unique key for the input combination
    unique_key = generate_unique_key(filename, prompt, reverse_prompt, num_inference_steps, guidance_scale, seed, re_p, re_b, ha_p, ga_b)
    print(f"unique_key: {unique_key} processed_results: {processed_results} ")
    if unique_key in processed_results:
        processed_filename = processed_results[unique_key]
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_filename}'}, broadcast=True, to=room_id)
        return

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    big_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], big_mask_filename)
    replace_path = os.path.join(app.config['UPLOAD_FOLDER'], replace_filename)
    if re_mask != 'F' or not os.path.exists(big_mask_path):
        print(f"big_mask_path: {big_mask_path} ")
        re_auto_mask_with_out_head_b(filename, int(re_p), int(re_b), int(ha_p), int(ga_b))

    processor = CustomInpaintPipeline()
    # Generate image with progress updates
    big_mask = Image.open(big_mask_path).convert("L")
    # Save processed image
    def progress_callback(step, t, latents):
        progress = int((step / num_inference_steps) * 100)
        emit('processing_progress', {'progress': progress}, broadcast=True, to=room_id)
    image_org = Image.open(file_path)
    original_width, original_height = image_org.size
    image = image_org.convert("RGB")
    # model_instance = LaMaModel(model_path='/mnt/sessd/ai_tools/Inpaint-Anything/big-lama')
    try:
        with torch.no_grad():
            result = processor.pipe(
                prompt = prompt,
                reverse_prompt= reverse_prompt,
                image=image,
                mask_image=big_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback=progress_callback,
                callback_steps=1  # 调用回调的步数
            ).images[0]
        result = result.resize((original_width, original_height), Image.LANCZOS)
        processed_r_filename = f'processed_r_{unique_key}.png'
        processed_r_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_r_filename)
        result.save(processed_r_file_path)
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_r_filename}'}, broadcast=True, to=room_id)

        blurred_mask = Image.open(replace_path).convert("L")

        fix_processed_filename =  f'fix_processed_it_{unique_key}.png'
        fix_processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], fix_processed_filename)
        blended_result = Image.composite(result, image, blurred_mask)
        # blended_result = model_instance.inpaint_and_blend(result, image, blurred_mask)
        blended_result.save(fix_processed_file_path)
        processed_results[unique_key] = fix_processed_filename
        emit('processing_done', {'processed_image_url': f'/uploads/{fix_processed_filename}'}, broadcast=True, to=room_id)
    finally:
        # 释放 GPU 内存
        del result
        del image
        del big_mask
        torch.cuda.empty_cache()
        gc.collect()


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, allow_unsafe_werkzeug=True)