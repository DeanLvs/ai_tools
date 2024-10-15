import os,json
import torch
from downloadC import download_file_c
import subprocess
from CoreProService import resize_image, generate_unique_key, re_auto_mask_with_out_head_b, GenContextParam, GenSavePathParam
import time
import threading
import threading
import queue
from CoreDb import RoomImageManager
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from PIL import Image,ImageDraw
from ImageProcessorSDXCN import CustomInpaintPipeline as sdxlInpainting
import base64
import traceback
import numpy as np
import requests
import gc
# 创建一个线程锁
lock = threading.Lock()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
socketio = SocketIO(app)
# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Dictionary to store processed results
processed_results = {}
processed_user_results={}
stop_it_set={}

task_positions = {}  # 用来记录 room_id 和队列中的位置
sdxl_ac_in_cpu = None
sdxl_ac_in_gpu = None
import shutil
import cv2

# 创建实例
room_image_manager = RoomImageManager('gen_room_t_images_kr.db')

glob_parts_to_include = None
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    room_id = request.form.get('room_id')
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id)
        # 创建房间
        os.makedirs(file_path, exist_ok=True)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, filename)

        file.save(file_path)
        # 备份原始图像
        backup_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, f"backup_{filename}")
        shutil.copy(file_path, backup_path)
        # 调整图像大小
        filename = resize_image(file_path)
        return jsonify({'filename': filename, 'file_url': f'{filename}'})

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['roomId']
    session_id = request.sid
    join_room(room_id)
    print(f'User {session_id} joined room {room_id}')

@socketio.on('re_get')
def handle_join_room(data):
    room_id = data['roomId']

    if room_id in task_positions:
        now_pas = task_positions[room_id]
        if now_pas > 1:
            emit('processing_step_progress', { 'text': f'book yes 你还有任务没有执行完成哈，当前排在{now_pas}位，等不及可晚点点击查看历史...'}, broadcast=True, to=room_id)
        else:
            emit('processing_step_progress',{'text': f'book yes 你还有任务没有执行完成哈，等不及可晚点点击查看历史...'}, broadcast=True, to=room_id)
    else:
        emit('processing_step_progress', {
            'text': f'book yes 少年...'}, broadcast=True, to=room_id)
        emit('processing_done_text', {
            'text': f'book yes 少年...'}, broadcast=True, to=room_id)
    imgStrList = room_image_manager.get_imgStrList(room_id)
    for imgStr, img_type, name, created_at in imgStrList:
        emit('processing_done', {'processed_image_url': f'{imgStr}' ,'img_type': img_type, 'name': name, 'keephide':'k'}, broadcast=True, to=room_id)
    emit('processing_step_fin', {'fin': 'f'}, broadcast=True, to=room_id)
    print(f're_get room {room_id}')
    total_users = len(socketio.server.manager.rooms['/'])
    print(f'total_users room {total_users}')
    # 向所有客户端广播在线人数
    emit('online_users', {'count': total_users}, broadcast=True)


@socketio.on('join')
def handle_join(data):
    room_id = data['roomId']
    # 将客户端的socket.id加入到指定的房间
    socketio.enter_room(data['sid'], room_id)
    print(f'User {data["sid"]} joined room {room_id}')
    total_users = len(socketio.server.manager.rooms['/'])
    # 向所有客户端广播在线人数
    emit('online_users', {'count': total_users}, broadcast=True)

@socketio.on('stop_de')
def handle_join(data):
    room_id = data['roomId']
    stop_it_set[room_id]=True

# 连接时自动加入一个以 session ID 为房间名的房间
@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    join_room(session_id)
    # 获取当前在线的总人数
    total_users = len(socketio.server.manager.rooms['/'])
    # 向所有客户端广播在线人数
    emit('online_users', {'count': total_users}, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    leave_room(session_id)
    # 获取当前在线的总人数
    total_users = len(socketio.server.manager.rooms['/'])

    # 向所有客户端广播在线人数
    emit('online_users', {'count': total_users}, broadcast=True)
    print(f'User {session_id} disconnected. Total users: {total_users}')

def run_gen_it(processor, g_c_param: GenContextParam):
    result_next,gen_it_prompt = processor.genIt(g_c_param)
    result_next = result_next.resize((g_c_param.original_width, g_c_param.original_height), Image.LANCZOS)
    processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], g_c_param.room_id, g_c_param.save_filename)
    result_next.save(processed_file_path)
    room_image_manager.insert_imgStr(g_c_param.room_id, f'{g_c_param.save_filename}', 'done', g_c_param.book_img_name, ext_info = gen_it_prompt, socketio=socketio)
    socketio.emit('processing_step_fin', {'fin': 'f'}, to=g_c_param.room_id)
    del result_next

@socketio.on('process_set_lora')
def process_set_lora(data):
    print("got it process_set_lora")
    room_id = data['roomId']
    if 'lora_id' in data and data['lora_id'] == 'x':
        processor = sdxlInpainting()
        processor.release_resources()
        socketio.emit('processing_done_text', {'text': 'xxx'}, to=room_id)
        return
    res_text = '无参数请输出 c 的 l 或者 w'
    if 'lora_id' in data and data['lora_id'] != '':
        lora_id = data['lora_id']
        lora_id_a = lora_id.split('#')
        lora_id = lora_id_a[0]
        lora_wight = float(lora_id_a[1])
        de_name = lora_id_a[2]
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
                print(f'had down loaded this {lora_id} {output_file}')
        else:
            output_file = f"/mnt/fast/civitai-downloader/lora/{lora_id}"
        print(f'loading this {lora_id} {lora_wight} {de_name} {output_file}')
        processor = sdxlInpainting()
        if lora_wight < 0:
            processor.unload_lora_weights()
            lora_wight = 0 - lora_wight
        processor.load_lora_weights(lora_path = output_file, adapter_name=de_name, lora_scale= lora_wight)
        print(f'loaded this {lora_id} {lora_wight} {de_name} {output_file}')
        res_text = '完成' + output_file
    elif 'wei_id' in data and data['wei_id'] != '':
        wei_id = data['wei_id']
        processor = sdxlInpainting()
        wei_id_a = wei_id.split('#')
        load_part_list = []
        weights_path = f"/mnt/fast/civitai-downloader/{wei_id_a[0]}"
        for par in wei_id_a:
            load_part_list.append(par)
        # load_part_list = ['unet', 'vae', 't1', 't2']
        processor.load_pretrained_weights(weights_path, load_part_list)
        res_text = '完成'+weights_path
    emit('processing_done_text', {'text': res_text}, broadcast=True, to=room_id)


def extract_data(data, fields):
    """
    通用数据提取方法，从 data 中提取指定的字段并设置默认值。
    fields 是一个字典，键为字段名，值为默认值。
    """
    return {field: data.get(field, default) for field, default in fields.items()}


def process_req_data(data):
    """
    处理图片相关的数据，使用通用方法提取字段，并对特定字段进行额外处理。
    """
    # 所有需要提取的字段及其默认值
    fields = {
        'def_skin': '',
        'filename': '',
        'prompt': '',
        'prompt_2': '',
        're_mask': '',
        're_p': '[]',  # 默认字符串形式，后续使用 eval
        're_b': '[]',  # 默认字符串形式，后续使用 eval
        'ha_p': 0,  # 整数
        'ga_b': 0.0,  # 浮点数
        'reverse_prompt': '',
        'reverse_prompt_2': '',
        'strength': 1.0,  # 浮点数
        'num_inference_steps': 50,  # 整数
        'guidance_scale': 7.5,  # 浮点数
        'seed': 126,  # 默认值为 None
        'roomId': ''  # 房间ID
    }

    # 提取数据
    processed_data = extract_data(data, fields)

    # 特殊字段处理
    try:
        processed_data['re_p_float_array'] = eval(processed_data['re_p'])  # 将 re_p 转换为浮点数组
    except Exception:
        processed_data['re_p_float_array'] = []  # 处理异常，确保返回空列表

    try:
        processed_data['re_b_float_array'] = eval(processed_data['re_b'])  # 将 re_b 转换为浮点数组
    except Exception:
        processed_data['re_b_float_array'] = []  # 处理异常，确保返回空列表

    # 强制转换类型，确保数据格式正确
    processed_data['ha_p'] = int(processed_data.get('ha_p', 0))  # 确保 ha_p 为整数
    processed_data['ga_b'] = float(processed_data.get('ga_b', 0.0))  # 确保 ga_b 为浮点数
    processed_data['strength'] = float(processed_data.get('strength', 1.0))  # 确保 strength 为浮点数
    processed_data['num_inference_steps'] = int(processed_data.get('num_inference_steps', 50))  # 确保为整数
    processed_data['guidance_scale'] = float(processed_data.get('guidance_scale', 7.5))  # 浮点数
    processed_data['seed'] = int(processed_data.get('seed', 126))
    print(f"req map : {processed_data}")
    # 返回完整处理的数据
    return processed_data

def save_all(img_to_save, img_to_save_path):
    img_to_save.save(img_to_save_path)
@socketio.on('process_image_remve')
def process_image_remve(data):
    processor = sdxlInpainting()
    processor.unload_lora_weights()
def room_path(room_id):
    return os.path.join(app.config['UPLOAD_FOLDER'], room_id)

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
def i2v_processing(processed_data):
    url = "http://8.130.32.107:7860/generate-video/"
    filename = processed_data['filename']
    room_id = processed_data['roomId']
    i2v_image_pil_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, filename)
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
    i2v_video_pil_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, i2v_video_name)
    return_i2v_video_pil_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, return_i2v_video_name)
    # 保存生成的视频
    with open(i2v_video_pil_path, "wb") as f:
        f.write(response.content)
    make_video_quicktime_compatible(i2v_video_pil_path, return_i2v_video_pil_path, target_width, target_height)
    print("视频生成并保存成功！")
    return return_i2v_video_name

def get_from_local(pil_img, save_img_path, conver = None):
    if pil_img:
        return pil_img
    elif isinstance(save_img_path, str) and os.path.exists(save_img_path):
        if conver:
            return Image.open(save_img_path).convert(conver)
        else:
            return Image.open(save_img_path)
    return None

def handle_image_processing_b(data):
    # 处理并提取图片数据
    processed_data = process_req_data(data)
    room_id = processed_data['roomId']
    if processed_data['def_skin'] == '99':
        i2v_video_name = i2v_processing(processed_data)
        room_image_manager.insert_imgStr(room_id, f'{i2v_video_name}', 'done', 'book', socketio=socketio)
        socketio.emit('processing_step_fin', {'fin': 'f'}, to=room_id)
        return

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
    socketio.emit('processing_step_progress', {'text': '到你了少年，开始处理...'}, to=room_id)
    re_mask = processed_data['re_mask']
    if seed > -2:
        torch.manual_seed(seed)
    # Generate a unique key for the input combination
    unique_key = generate_unique_key(room_id, filename, prompt, reverse_prompt, num_inference_steps, guidance_scale, seed, re_p_float_array, re_b_float_array, ha_p, ga_b, strength,prompt_2, reverse_prompt_2)

    print(f"unique_key: {unique_key} processed_results: {processed_results} ")

    if unique_key in processed_results:
        processed_filename = processed_results[unique_key]
        socketio.emit('processing_step_progress', {'text': 'book yes 已出完请点击查看历史产看...'}, to=room_id)
        return
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, filename)

    mask_future = None
    gen_fix_pic_future = None

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

    gen_save_path = GenSavePathParam(app.config['UPLOAD_FOLDER'], room_id, filename)

    f_output_line_image_pil = None
    if re_mask == 'F':
        print(f"filled_image_pil_filename: {gen_save_path.filled_image_pil_filename} ")
        socketio.emit('processing_step_progress', {'text': 'book yes 识别图片...'},to=room_id)
        room_upload_folder = room_path(room_id)
        mask_future, gen_fix_pic_future, segmentation_image_pil = re_auto_mask_with_out_head_b(room_upload_folder, filename)

    if os.path.exists(gen_save_path.line_file_name_path):
        print(f'open {gen_save_path.line_file_name_path} use it line page')
        line_image_l_control = Image.open(gen_save_path.line_file_name_path)
        print(f'control_image is {type(line_image_l_control)}')

    image_org = Image.open(file_path).convert("RGB")
    original_width, original_height = image_org.size
    orgImage = image_org

    if segmentation_image_pil:
        room_image_manager.insert_imgStr(room_id, f'{gen_save_path.mask_clear_finally_filename}', 'pre_done', '分割图', 'k',
                  segmentation_image_pil, gen_save_path.segmentation_image_pil_path, socketio=socketio)
    segmentation_image_pil = get_from_local(segmentation_image_pil, gen_save_path.segmentation_image_pil_path)

    def progress_callback(step, t, latents, had):
        progress = int((step / num_inference_steps) * 100)
        socketio.emit('processing_progress', {'progress': progress, 'had': had}, to=room_id)
        if room_id in stop_it_set:
            stop_it_set.remove(room_id)
            socketio.emit('processing_progress', {'progress': 'stop', 'had': had}, to=room_id)
            return True

    # 定义回调函数并传递 'had' 参数
    def create_callback(had_value):
        return lambda step, t, latents: progress_callback(step, t, latents, had_value)

    global sdxl_ac_in_cpu
    global sdxl_ac_in_gpu
    processor = None
    try:
        if mask_future is not None:
            mask_clear_finally, temp_humanMask, control_image_return, normal_map_img_return, normal_3d_map_img_return, with_torso_mask, f_output_line_image_pil = mask_future.result()
            room_image_manager.insert_imgStr(room_id, f'{gen_save_path.mask_clear_finally_filename}', 'pre_done', '重绘区域', 'k', mask_clear_finally, gen_save_path.mask_clear_finally_path, socketio=socketio)
            # if ha_p > 0:
            room_image_manager.insert_imgStr(room_id, f'{gen_save_path.control_net_fname}', 'pre_done', '姿势识别', 'k', control_image_return, gen_save_path.control_net_fname_path, socketio=socketio)
            if normal_map_img_return:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.nor_control_net_fname}', 'pre_done', '深度识别', 'k', normal_map_img_return, gen_save_path.nor_control_net_fname_path, socketio=socketio)
            if normal_3d_map_img_return:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.nor_3d_control_net_fname}', 'pre_done', '3d深度识别', 'k', normal_3d_map_img_return, gen_save_path.nor_3d_control_net_fname_path, socketio=socketio)
            if with_torso_mask:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.with_torso_mask_name}', 'pre_done', '最终重绘区域', 'k', with_torso_mask, gen_save_path.with_torso_mask_name_path, socketio=socketio)
            if f_output_line_image_pil:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.line_file_auto_name}', 'pre_done', '排除头部手臂的边线', 'k', f_output_line_image_pil, gen_save_path.line_file_auto_name_path, socketio=socketio)

        socketio.emit('processing_step_progress', {'text': 'book yes 加载模型中...'}, to=room_id)
        # processor =FluxInpaintPipelineSingleton ()
        print('------------开始加载 vae unet ---------------')
        processor = sdxlInpainting(controlnet_list_t = re_p_float_array, use_out_test= (ga_b == 1.0))
        if sdxl_ac_in_cpu:
            print(f'--------------从 cpu加载模型到 gpu')
            torch.cuda.empty_cache()
            gc.collect()
            sdxl_ac_in_gpu = processor.pipe.unet.to('cuda')
            print(f'--------------完成 cpu加载模型到 gpu')
            sdxl_ac_in_cpu_del = sdxl_ac_in_cpu
            sdxl_ac_in_cpu = None
            del sdxl_ac_in_cpu_del
            torch.cuda.empty_cache()
            gc.collect()
            print(f'--------------完成 释放 cpu空间-----------')
        else:
            sdxl_ac_in_gpu = processor.pipe.unet
        socketio.emit('processing_step_progress', {'text': 'book yes 计算中 共4张图可出稍等哈...'}, to=room_id)

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

        big_mask = get_from_local(mask_clear_finally, gen_save_path.mask_clear_finally_path, "L")
        normal_map_img = get_from_local(normal_map_img_return, gen_save_path.nor_control_net_fname_path)
        normal_map_3d_img = get_from_local(normal_3d_map_img_return, gen_save_path.nor_3d_control_net_fname_path)
        g_c_param = GenContextParam(prompt, prompt_2, reverse_prompt, reverse_prompt_2, orgImage, with_torso_mask_img,
                   num_inference_steps, guidance_scale,
                   create_callback('1/4'), strength,
                   [control_image, normal_map_img, segmentation_image_pil, line_image_l_control], re_b_float_array,
                   prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
                   original_width, original_height, '', room_id)
        g_c_param.next_genImage, g_c_param.save_filename = orgImage, f'p_1_{unique_key}.png'
        g_c_param.book_img_name = '原始生成'
        run_gen_it(processor, g_c_param)

        if gen_fix_pic_future is not None:
            clear_image_pil_return, filled_image_pil_return, next_filled_image_pil_return = gen_fix_pic_future.result()
            room_image_manager.insert_imgStr(room_id, f'{gen_save_path.cur_res_filename}', 'pre_done', '清除后图片', None, clear_image_pil_return, gen_save_path.cur_res_path, socketio=socketio)
            if filled_image_pil_return:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.filled_image_pil_filename}', 'pre_done', '自动识别填充' ,None,filled_image_pil_return, gen_save_path.filled_image_pil_path, socketio=socketio)
            if isinstance(next_filled_image_pil_return, Image.Image) and next_filled_image_pil_return is not None:
                room_image_manager.insert_imgStr(room_id, f'{gen_save_path.next_filled_image_pil_filename}', 'pre_done', '默认填充', None, next_filled_image_pil_return, gen_save_path.next_filled_image_pil_path, socketio=socketio)

        clear_org_i = get_from_local(clear_image_pil_return, gen_save_path.cur_res_path)
        g_c_param.func_call_back, g_c_param.next_genImage, g_c_param.save_filename = create_callback('2/4'), clear_org_i,  f'p_2_{unique_key}.png'
        g_c_param.book_img_name = '擦出生成'
        run_gen_it(processor, g_c_param)

        genImage = get_from_local(filled_image_pil_return, gen_save_path.filled_image_pil_path)

        if isinstance(genImage, Image.Image) and genImage is not None:
            g_c_param.func_call_back, g_c_param.next_genImage, g_c_param.save_filename = create_callback('3/4'), genImage,  f'p_2_{unique_key}.png'
            g_c_param.book_img_name = '填充纹理'
            run_gen_it(processor, g_c_param)

        if auto_line_con:
            if normal_map_3d_img is not None:
                re_b_float_array[1], g_c_param.control_img_list[1] = re_b_float_array[2], None #normal_map_3d_img
            else:
                print("did not set 3d nor------")
            g_c_param.control_img_list[3] = auto_line_con
            print("use auto line con ------")
            g_c_param.func_call_back , g_c_param.next_genImage, g_c_param.save_filename = create_callback('4/4'), genImage,  f'p_4_{unique_key}.png'
            g_c_param.book_img_name = '预测遮挡'
            run_gen_it(processor, g_c_param)
        next_genImage = get_from_local(next_filled_image_pil_return, gen_save_path.next_filled_image_pil_path)
        # if next_genImage 后 可以执行 默认填充 g_c_param.next_genImage = next_genImage 可以执行 g_c_param.save_filename = processed_r_r_filename
        processed_results[unique_key] = unique_key
    except Exception as e:
        print(f"processing error is  -------: {e}")
        socketio.emit('processing_step_progress', {'text': 'book yes 异常了请重新上传执行...'}, to=room_id)
    finally:
        print(f'--------------推理完成开始 释放 gpu空间-----------')
        torch.cuda.empty_cache()
        gc.collect()
        print(f'--------------释放 gpu空间 完成 开始移动 模型到 cpu-----------')
        sdxl_ac_in_cpu = processor.pipe.unet.to("cpu")
        print(f'--------------完成移动 模型到 cpu---开始释放GPU资源----------')
        sdxl_ac_in_gpu_del = sdxl_ac_in_gpu
        sdxl_ac_in_gpu = None
        del sdxl_ac_in_gpu_del
        torch.cuda.empty_cache()
        gc.collect()
        print(f'--------------完成移动 模型到 cpu---完成释放GPU资源----------')
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

@socketio.on('process_replace_image_b')
def process_replace_image_b(data):
    print(f'get it process_replace_image_b')
    # 读取图像
    filename_face = data['filename_face']
    filename_handler = data['filename_handler']
    room_id = data['room_id']
    path = room_path(room_id)
    print(f'get it process_replace_image_b {filename_face} {filename_handler} {room_id} {path}')
    img_face = cv2.imread(os.path.join(path, filename_face))

    img_handler = cv2.imread(os.path.join(path, filename_handler))

    # 将图像编码为 JPEG 格式
    _, img_encoded_face = cv2.imencode('.png', img_face)

    _, img_encoded_handler = cv2.imencode('.png', img_handler)

    img_base64_face = base64.b64encode(img_encoded_face).decode('utf-8')
    img_base64_handler = base64.b64encode(img_encoded_handler).decode('utf-8')
    # 发送图像数据给服务器
    data = {'image_data': img_base64_face, 'image_data_n': img_base64_handler}
    response = requests.post('http://localhost:5000/process_image', json=data)

    # 获取返回的处理后的图像数据
    processed_image = response.json()['processed_image']

    # 将 Base64 字符串解码为二进制数据
    img_binary = base64.b64decode(processed_image)
    # 将二进制数据转换为 NumPy 数组并解码为图像
    img_np = np.frombuffer(img_binary, np.uint8)
    processed_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    re_fina = 'finaly_'+filename_handler
    print(f'get it {re_fina}')
    re_fina_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, re_fina)
    print(f'start save it {re_fina_path}')
    cv2.imwrite(re_fina_path, processed_img)
    print(f'suc save it {re_fina}')
    room_image_manager.insert_imgStr(room_id, f'{re_fina}', 'done', 'book', socketio=socketio)
    socketio.emit('processing_step_fin', {'fin': 'f'}, to=room_id)

@app.route('/api/rep_upload_face', methods=['POST'])
def rep_upload_face():
    if 'file_face_img' not in request.files:
        return 'No file part'
    file_face_img = request.files['file_face_img']
    room_id = request.form.get('room_id')
    if file_face_img.filename == '':
        return 'No selected file'
    if file_face_img:
        file_face_img_filename = 'r_f_'+file_face_img.filename
        file_face_img_file_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, file_face_img_filename)
        file_face_img.save(file_face_img_file_path)
        # 调整图像大小
        file_face_img_filename = resize_image(file_face_img_file_path)
        return jsonify({'filename_face': file_face_img_filename, 'file_face_url': f'{file_face_img_filename}'})


@app.route('/uploads/<room_id>/<filename>')
def uploaded_file(room_id, filename):
    room_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id)
    return send_from_directory(room_path, filename)
# 路由来提供你的 CSS 文件
@app.route('/css/<filename>')
def send_css(filename):
    return send_from_directory('/mnt/sessd/ai_tools/templates/css', filename)
# 路由来提供你的 CSS 文件
@app.route('/js/<filename>')
def send_js(filename):
    return send_from_directory('/mnt/sessd/ai_tools/templates/js', filename)


# 创建队列，最大容量为30
task_queue = queue.Queue(maxsize=10)

@socketio.on('process_image_b')
def add_image_processing_task(data):
    room_id = data['roomId']
    with lock:
        try:
            if room_id in task_positions:
                emit('processing_step_progress', {'text': f'book yes 你还有任务在执行哈，点击查看历史 看看进度吧...'}, broadcast=True, to=room_id)
            else:
                # 尝试将任务放入队列，如果队列已满，抛出异常
                task_queue.put(data, block=False)
                position_in_queue = task_queue.qsize()
                task_positions[room_id] = position_in_queue
                if position_in_queue >= 2:
                    emit('processing_step_progress', {'text': f'book yes 已经进入执行队列，当前排在{position_in_queue}位，已经记录您的提交，等不及可晚点点击查看历史...'}, broadcast=True, to=room_id)
                else:
                    emit('processing_step_progress',
                         {'text': f'book yes 已经进入执行队列，当前排在{position_in_queue}位，这就开始执行...'},broadcast=True, to=room_id)
        except queue.Full:
            if room_id in task_positions:
                print(f'失败清理排队位置了')
                del task_positions[room_id]
            # 队列已满，返回错误
            emit('processing_step_progress', {'text': 'book yes 太火爆已满，稍后在线人数较少时重新提交可好...'}, broadcast=True, to=room_id)
# 同步更新 task_positions 中任务的队列位置
def update_task_positions(finished_room_id):
    # 遍历字典，减小所有 room_id 的位置
    for room_id in task_positions:
        if task_positions[room_id] > task_positions.get(finished_room_id, 0):
            task_positions[room_id] -= 1
# 消费者线程，持续从队列中获取任务进行处理
def queue_consumer():
    while True:
        # 从队列中获取任务
        task_data = task_queue.get()
        room_id = task_data.get('roomId')
        if task_data is None:
            break
        try:
            handle_image_processing_b(task_data)
        except Exception as e:
            print(f"task error: {e}")
            traceback.print_exc()
        finally:
            task_queue.task_done()
            # 任务已被消费，移除它的 room_id
            if room_id in task_positions:
                del task_positions[room_id]
                # 同步更新剩余任务的位置
            update_task_positions(room_id)

# 启动消费者线程
consumer_thread = threading.Thread(target=queue_consumer, daemon=True)
consumer_thread.start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, allow_unsafe_werkzeug=True)