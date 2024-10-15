import os
import torch
import numpy as np
from downloadC import download_file_c
import hashlib
import time
import traceback
import scipy.ndimage
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from PIL import ImageDraw,Image, ImageFilter
from ImageProcessorSDX import CustomInpaintPipeline
from mask_generator import MaskGenerator  # 导入 MaskGenerator 类
import gc
from concurrent.futures import ThreadPoolExecutor
import cv2
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
socketio = SocketIO(app)
# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Dictionary to store processed results
processed_results = {}
from sklearn.cluster import KMeans
processed_user_results={}
@app.route('/')
def index():
    return render_template('index.html')

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
            # filled_image_pil.save(fix_gen_path)
            print(f'finish check pose an ge wenli....')
            return clear_result, filled_image_pil, mask_clear
        else:
            print("error LaMaModel")
    except Exception as ex:
        print(ex)
        # 打印完整的异常堆栈
        traceback.print_exc()
@app.route('/generate_mask', methods=['POST'])
def generate_mask():
    data = request.json
    # filename = data['filename']
    # points = data['points']
    # points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]  # 将点转换为坐标对
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #
    # image = Image.open(file_path).convert("RGB")
    # mask = Image.new("L", image.size, 0)
    # draw = ImageDraw.Draw(mask)
    # for point in points:
    #     draw.point(point, fill=255)
    # mask_np = np.array(mask)
    # replace_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
    # mask.save(replace_mask_path)
    # # Save big mask
    # big_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'big_mask_' + filename)
    # mask.save(big_mask_path)
    # gen_fix_pic(file_path, image, mask, mask_np, file_path, filename)
    # return jsonify({'mask_url': f'/uploads/big_mask_' + filename})

@app.route('/auto_recognize', methods=['POST'])
def auto_recognize():
    data = request.json
    filename = data['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # # 调用 re_auto_mask 获取合并后的掩码
    # merged_mask,close_mask_img, not_close_mask= re_auto_mask(file_path)
    # # 获取合并掩码的像素位置
    # merged_mask_pixels = np.argwhere(merged_mask > 0)
    # # 转换为原生 Python int 类型
    # merged_mask_pixels = [[int(x), int(y)] for y, x in merged_mask_pixels]
    # return jsonify({'mask_pixels': merged_mask_pixels})

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

def re_auto_mask_with_out_head_b(filename, re_p=30, re_b=0, ha_p=30, ga_b=0, re_mask = 'F', extracted_textures = None):
    print(f"  re_p: {re_p} re_b: {re_b} or ha_p: {ha_p}  or ga_b: {ga_b} ")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_pil = Image.open(file_path).convert("RGB")
    remove_and_rege_mask, close_mask_img, not_close_mask_img = re_auto_mask(file_path)
    merged_mask_pixels = np.argwhere(remove_and_rege_mask > 0)
    merged_mask_pixels = [[int(x), int(y)] for y, x in merged_mask_pixels]
    mask_clear = Image.new("L", image_pil.size, 0)
    draw = ImageDraw.Draw(mask_clear)
    for point in merged_mask_pixels:
        draw.point(point, fill=255)
    print(f"finish draw close mask")
    clear_result, filled_image_pil, mask_clear_finally = gen_fix_pic(file_path, image_pil, mask_clear,
                                                                     not_close_mask_img, extracted_textures)
    print(f"finish trans wenli an clear")
    return mask_clear, clear_result, filled_image_pil, mask_clear_finally


@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['roomId']
    session_id = request.sid
    join_room(room_id)
    print(f'User {session_id} joined room {room_id}')

@socketio.on('re_get')
def handle_join_room(data):
    room_id = data['roomId']
    if room_id in processed_user_results:
        imgStrList = processed_user_results[room_id]
        for imgStr in imgStrList:
            emit('processing_done', {'processed_image_url': f'{imgStr}','keephide':'k'}, broadcast=True, to=room_id)
    print(f're_get room {room_id}')


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

def genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, genImage, big_mask,
          num_inference_steps, guidance_scale, progress_callback, strength=0.8):
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
@socketio.on('process_set_lora')
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
                        chunk_start_time = time.time()
                        buffer = response.read(CHUNK_SIZE)
                        chunk_end_time = time.time()

                        if not buffer:
                            break

                        downloaded += len(buffer)
                        f.write(buffer)
                        chunk_time = chunk_end_time - chunk_start_time

                        if chunk_time > 0:
                            speed = len(buffer) / chunk_time / (1024 ** 2)

                        if total_size is not None:
                            progress = downloaded / total_size
                            emit('processing_progress', {'progress': progress}, broadcast=True, to=room_id)
                            # sys.stdout.write(f'\rDownloading: {filename} [{progress * 100:.2f}%] - {speed:.2f} MB/s')
                            # sys.stdout.flush()
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
        res_text = '完成'+weights_path
    emit('processing_done_text', {'text': res_text}, broadcast=True, to=room_id)



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
@socketio.on('process_image_b')
def handle_image_processing_b(data):
    print("got it process_image_b")
    def_skin = str(data['def_skin'])
    filename = data['filename']
    prompt = data['prompt']
    prompt_2 = data['prompt_2']
    re_mask = data['re_mask']
    re_p = data['re_p']
    re_b = int(data['re_b'])
    ha_p = data['ha_p']
    ga_b = data['ga_b']
    reverse_prompt = data['reverse_prompt']
    reverse_prompt_2 = data['reverse_prompt_2']
    strength = float(data['strength'])
    num_inference_steps = data['num_inference_steps']
    guidance_scale = data['guidance_scale']
    seed = data['seed']
    room_id = None
    if 'roomId' in data:
        room_id = data['roomId']

    if def_skin == '1':
        extracted_textures = get_def_extracted_textures()
    else:
        extracted_textures = None
    imgStrList = []
    if room_id:
        if room_id in processed_user_results:
            imgStrList = processed_user_results[room_id]
        else:
            processed_user_results[room_id] = imgStrList
    if not strength:
        strength = 0.8
    if not prompt or prompt == "":
        prompt = None #consistent angles, original pose,seamless background integration, matching lighting and shading, matching skin tone, no deformation
    if not prompt_2 or prompt_2 == "":
        prompt_2 = None
    if not reverse_prompt or reverse_prompt == "":
        reverse_prompt = None
    if not reverse_prompt_2 or reverse_prompt_2 == "":
        reverse_prompt_2 = None
    if not num_inference_steps:
        num_inference_steps = 120
    else:
        num_inference_steps = int(num_inference_steps)
    if not guidance_scale:
        guidance_scale = 8.7
    else:
        guidance_scale = float(guidance_scale)
    # 设置随机种子
    if not seed:
        seed = 56
    else:
        seed = int(seed)
    if seed > -2:
        torch.manual_seed(seed)
        print(f"prompt: {prompt} reverse_prompt: {reverse_prompt} "
          f"num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}, "
          f"seed: {seed} strength: {strength} prompt_2: {prompt_2} reverse_prompt_2: {reverse_prompt_2}")
    else:
        print(f"prompt: {prompt} reverse_prompt: {reverse_prompt} "
              f"num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale} strength: {strength} prompt_2: {prompt_2} reverse_prompt_2: {reverse_prompt_2}")

    # Generate a unique key for the input combination
    unique_key = generate_unique_key(filename, prompt, reverse_prompt, num_inference_steps, guidance_scale, seed, re_p, re_b, ha_p, ga_b, strength,prompt_2, reverse_prompt_2)
    print(f"unique_key: {unique_key} processed_results: {processed_results} ")
    if unique_key in processed_results:
        processed_filename = processed_results[unique_key]
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_filename}'}, broadcast=True, to=room_id)
        return
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mask_clear_finally = None
    print(f"re_mask: {re_mask} ")

    filled_image_pil_return = None
    clear_return = None
    mask_clear_path_filename = 'mask_clear_return_' + filename
    mask_clear_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_clear_path_filename)

    filled_image_pil_filename = 'filled_image_pil_' + filename
    filled_image_pil_path = os.path.join(app.config['UPLOAD_FOLDER'], filled_image_pil_filename)

    cur_res_filename = 'clear_return_' + filename
    cur_res_path = os.path.join(app.config['UPLOAD_FOLDER'], cur_res_filename)

    mask_clear_finally_filename = 'filled_use_mask_image_pil_' + filename
    mask_clear_finally_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_clear_finally_filename)

    if re_mask != 'F' or not os.path.exists(mask_clear_path):
        print(f"mask_clear_path: {mask_clear_path} ")
        mask_clear_return, clear_return, filled_image_pil_return, mask_clear_finally = re_auto_mask_with_out_head_b(filename, int(re_p), int(re_b), int(ha_p), int(ga_b), re_mask, extracted_textures=extracted_textures)
        mask_clear_return.save(mask_clear_path)

        cv2.imwrite(cur_res_path, clear_return)
        filled_image_pil_return.save(filled_image_pil_path)
        mask_clear_finally.save(mask_clear_finally_path)

        emit('processing_done', {'processed_image_url': f'/uploads/{mask_clear_finally_filename}'}, broadcast=True, to=room_id)
        imgStrList.append(f'/uploads/{mask_clear_finally_filename}')

        emit('processing_done', {'processed_image_url': f'/uploads/{cur_res_filename}'}, broadcast=True, to=room_id)
        imgStrList.append(f'/uploads/{cur_res_filename}')

        emit('processing_done', {'processed_image_url': f'/uploads/{filled_image_pil_filename}'}, broadcast=True, to=room_id)
        imgStrList.append(f'/uploads/{filled_image_pil_filename}')
    emit('processing_done_text', {'text': '正在book yes(不可言说，加载模型中，再次点击获取历史结果)'}, broadcast=True, to=room_id)
    # Generate image with progress updates
    # Save processed image
    def progress_callback(step, t, latents):
        progress = int((step / num_inference_steps) * 100)
        emit('processing_progress', {'progress': progress}, broadcast=True, to=room_id)
    image_org = Image.open(file_path)
    original_width, original_height = image_org.size
    orgImage = image_org.convert("RGB")

    if filled_image_pil_return:
        genImage = filled_image_pil_return
    elif isinstance(filled_image_pil_path, str) and os.path.exists(filled_image_pil_path):
        genImage = Image.open(filled_image_pil_path).convert("RGB")

    if mask_clear_finally:
        big_mask = mask_clear_finally
    elif isinstance(mask_clear_finally_path, str) and os.path.exists(mask_clear_finally_path):
        big_mask = Image.open(mask_clear_finally_path).convert("L")

    if isinstance(clear_return, np.ndarray) and clear_return is not None and clear_return.any():
        # 将 BGR 转换为 RGB
        clear_return_rgb_image = cv2.cvtColor(clear_return, cv2.COLOR_BGR2RGB)

        # 将转换后的 NumPy 数组 (RGB) 转换为 PIL.Image 对象
        clear_org_i = Image.fromarray(clear_return_rgb_image)
    elif isinstance(cur_res_path, str) and os.path.exists(cur_res_path):
        clear_org_i = Image.open(cur_res_path).convert("RGB")

    processor = CustomInpaintPipeline()
    try:
        orgResult = genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, orgImage, big_mask, num_inference_steps, guidance_scale,
                          progress_callback, strength)
        orgResult = orgResult.resize((original_width, original_height), Image.LANCZOS)
        processed_org_r_filename = f'processed_org_r_{unique_key}.png'
        processed_org_r_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_org_r_filename)
        orgResult.save(processed_org_r_file_path)
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_org_r_filename}'}, broadcast=True,
             to=room_id)
        imgStrList.append(f'/uploads/{processed_org_r_filename}')
        del orgResult

        org_fix_Result = genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, clear_org_i, big_mask, num_inference_steps, guidance_scale,
                          progress_callback, strength)
        org_fix_Result = org_fix_Result.resize((original_width, original_height), Image.LANCZOS)
        clear_f_fname = f'processed_clear_r_{unique_key}_' + filename
        org_fix_Result.save(os.path.join(app.config['UPLOAD_FOLDER'], clear_f_fname))
        emit('processing_done', {'processed_image_url': f'/uploads/{clear_f_fname}'}, broadcast=True,
             to=room_id)
        imgStrList.append(f'/uploads/{clear_f_fname}')

        del org_fix_Result

        result = genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, genImage, big_mask, num_inference_steps, guidance_scale, progress_callback, strength)
        result = result.resize((original_width, original_height), Image.LANCZOS)
        processed_r_filename = f'processed_r_{unique_key}.png'
        processed_r_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_r_filename)
        result.save(processed_r_file_path)
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_r_filename}'}, broadcast=True, to=room_id)
        imgStrList.append(f'/uploads/{processed_r_filename}')

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