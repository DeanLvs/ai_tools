import os
import numpy as np
import hashlib
import traceback
import scipy.ndimage
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from PIL import ImageDraw,Image, ImageFilter
from mask_generator import MaskGenerator  # 导入 MaskGenerator 类
from concurrent.futures import ThreadPoolExecutor
import cv2
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
socketio = SocketIO(app)
# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Dictionary to store processed results
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
def dilate_mask_np(mask, iterations):
    kernel = np.ones((3, 3), np.uint8)
    fix_mask_dilated = cv2.dilate(mask, kernel, iterations=iterations)
    # 确保 fix_mask 是一个值在 0 和 255 之间的 uint8 类型数组
    fix_mask_dilated = np.clip(fix_mask_dilated, 0, 255).astype(np.uint8)
    return fix_mask_dilated


def dilate_mask_by_percentage(mask, percentage, max_iterations=50):
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
    kernel = np.ones((2, 2), np.uint8)
    dilated_mask = mask.copy()
    last_non_zero = initial_non_zero

    for i in range(max_iterations):
        # 扩张掩码
        dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=1)
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

def load_texture(texture_path):
    # 读取图像
    texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    if texture is None:
        raise ValueError(f"无法读取纹理图像：{texture_path}")

    # 检查是否具有Alpha通道
    if texture.shape[2] == 4:
        # 已有Alpha通道，设置为不透明
        texture[:, :, 3] = 255
    else:
        # 没有Alpha通道，添加Alpha通道并设置为不透明
        alpha_channel = np.full((texture.shape[0], texture.shape[1], 1), 255, dtype=np.uint8)
        texture = np.concatenate((texture, alpha_channel), axis=2)

    return texture
def gen_fix_pic(file_path, file_i, mask_i, mask_np, fix_gen_path, file_name = "clear"):
    import cv2
    try:
        texture = load_texture("/Users/dean/Desktop/56f3d272528601debb9ce029f2a8d302.png")
        print("start apply_skin_tone")
        from SupMaskSu import apply_skin_tone, handlerMaskAll, extract_texture, warp_texture , \
            find_non_empty_texture
        humanMask = handlerMaskAll(file_path)
        image_bgr = cv2.imread(file_path)

        extracted_textures = extract_texture(image_bgr, humanMask, mask_np)
        filled_image = image_bgr.copy()
        # 填充纹理
        for part, mask in humanMask.items():
            texture = extracted_textures.get(part, None)
            if texture is not None and np.count_nonzero(texture[:, :, 3]) > 0:
                filled_image = warp_texture(filled_image, mask, texture, mask_np)
            else:
                backup_texture = find_non_empty_texture(extracted_textures)
                if backup_texture is not None:
                    filled_image = warp_texture(filled_image, mask, backup_texture, mask_np)

        # 保存结果
        filled_image_pil = Image.fromarray(cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB))
        filled_image_pil.save("filled_image.png")
    except Exception as ex:
        print(ex)
        # 打印完整的异常堆栈
        traceback.print_exc()


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
    print(f"  auto mask: {file_path}")
    # 生成两个掩码
    mask_generator = MaskGenerator()
    image_pil = Image.open(file_path).convert("RGB")
    close_mask = mask_generator.generate_mask(image_pil)
    # 转换掩码为 ndarrays
    densepose_mask_ndarray = convert_to_ndarray(close_mask)
    densepose_mask_ndarray = dilate_mask_by_percentage(densepose_mask_ndarray, 4)
    return densepose_mask_ndarray

def re_auto_mask_with_out_head_b(filename, re_p=30, re_b=0, ha_p=30, ga_b=0, re_mask = 'F'):
    print(f"  re_p: {re_p} re_b: {re_b} or ha_p: {ha_p}  or ga_b: {ga_b} ")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_pil = Image.open(file_path).convert("RGB")
    remove_and_rege_mask = re_auto_mask(file_path)
    merged_mask_pixels = np.argwhere(remove_and_rege_mask > 0)
    merged_mask_pixels = [[int(x), int(y)] for y, x in merged_mask_pixels]
    mask_clear = Image.new("L", image_pil.size, 0)
    draw = ImageDraw.Draw(mask_clear)
    for point in merged_mask_pixels:
        draw.point(point, fill=255)
    mask_clear.save(os.path.join(app.config['UPLOAD_FOLDER'], 'final_remv_s_' + filename))
    file_path_fi = os.path.join(app.config['UPLOAD_FOLDER'], 'final_' + filename)
    cur_res = gen_fix_pic(file_path, image_pil, mask_clear, np.array(mask_clear), file_path_fi, filename)
    return cur_res


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
        prompt = "" #consistent angles, original pose,seamless background integration, matching lighting and shading, matching skin tone, no deformation
    reverse_prompt = data['reverse_prompt']
    if not reverse_prompt or reverse_prompt == "":
        reverse_prompt = None
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
    re_auto_mask_with_out_head_b(filename, int(re_p), int(re_b), int(ha_p), int(ga_b), re_mask)



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, allow_unsafe_werkzeug=True)