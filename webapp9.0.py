import os,json,uuid
from downloadC import download_file_c
from BookYesCommon import load_unfinished_tasks, User, get_rest_key, resize_image, add_task_list, glob_task_positions, video_task_positions, glob_task_queue, task_vide_queue
from CoreService import handle_image_processing_b
import time
import threading
from CoreDb import RoomImageManager, user_states_control, query_or_def, user_vip_control
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from ImageProcessorSDXCN import CustomInpaintPipeline as sdxlInpainting
import base64
import traceback
import numpy as np
import requests
from TelRoboApp import queue_tel_consumer, run_telegram_bot, notify_it as tel_notify_it
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/nvme0n1-disk/book_yes/static/uploads'
app_path = app.config['UPLOAD_FOLDER']
socketio = SocketIO(app)
# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Dictionary to store processed results
room_image_manager = RoomImageManager()
stop_it_set={}

import shutil
import cv2
from book_yes_logger_config import logger
import inspect, asyncio

@app.route('/')
def index():
    return render_template('index.html')

def notify_it(iswct='ws', event=None, js_obj=None, to=None, keyList=None, media_group=None):
    if iswct == 'ws':
        socketio.emit(event, js_obj, to=to)
    elif iswct == 'tel':
        logger.info('not notify_it------had tel-----------')
# 定义方法映射
function_mapping = {
    'ws_notify_it': notify_it,
    'tel_notify_it': tel_notify_it,
    # 你可以在这里添加更多可用的函数
}

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
        room_image_manager.insert_imgStr(room_id, f'{filename}', 'done',
                                         '原图', ext_info='', notify_fuc=notify_it,
                                         notify_type='ws')
        return jsonify({'filename': filename, 'file_url': f'{filename}'})

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['roomId']
    session_id = request.sid
    join_room(room_id)
    logger.info(f'User {session_id} joined room {room_id}')

@socketio.on('re_get')
def handle_join_room(data):
    room_id = data['roomId']
    data['notify_type'] = 'ws'
    notify_type = data['notify_type']
    if room_id in glob_task_positions:
        now_pas = glob_task_positions[room_id]
        if now_pas > 1:
            notify_it(notify_type, 'processing_step_progress', { 'text': f'book yes 你还有任务没有执行完成哈，当前排在{now_pas}位...'}, to=room_id)
        else:
            notify_it(notify_type, 'processing_step_progress',{'text': f'book yes 你还有任务没有执行完成哈，正在处理，处理完成后可上传第二张...'}, to=room_id)
    else:
        notify_it(notify_type, 'processing_step_progress', {
            'text': f'book yes 处理中...'}, to=room_id)
        notify_it(notify_type, 'processing_done_text', {
            'text': f'book yes 处理中...'}, to=room_id)
    imgStrList = room_image_manager.get_imgStrList(room_id)
    for imgStr, img_type, name, created_at in imgStrList:
        notify_it(notify_type, 'processing_done', {'processed_image_url': f'{imgStr}' ,'img_type': img_type, 'name': name, 'keephide':'k'}, to=room_id)
    notify_it(notify_type, 'processing_step_fin', {'fin': 'f'},  to=room_id)
    logger.info(f're_get room {room_id}')
    total_users = len(socketio.server.manager.rooms['/'])
    logger.info(f'total_users room {total_users}')
    # 向所有客户端广播在线人数
    emit('online_users', {'count': total_users}, broadcast=True)


@socketio.on('join')
def handle_join(data):
    room_id = data['roomId']
    # 将客户端的socket.id加入到指定的房间
    socketio.enter_room(data['sid'], room_id)
    logger.info(f'User {data["sid"]} joined room {room_id}')
    total_users = len(socketio.server.manager.rooms['/'])
    # 向所有客户端广播在线人数
    emit( 'online_users', {'count': total_users}, broadcast=True)

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
    emit( 'online_users', {'count': total_users}, broadcast=True)
    logger.info(f'User {session_id} disconnected. Total users: {total_users}')


@socketio.on('process_set_lora')
def process_set_lora(data):
    logger.info("got it process_set_lora")
    room_id = data['roomId']
    if 'lora_id' in data and data['lora_id'] == 'x':
        processor = sdxlInpainting()
        processor.release_resources()
        notify_it('ws', 'processing_done_text', {'text': 'xxx'}, to=room_id)
        return
    res_text = '无参数请输出 c 的 l 或者 w'
    if 'lora_id' in data and data['lora_id'] != '':
        lora_id = data['lora_id']
        lora_id_a = lora_id.split('#')
        lora_id = lora_id_a[0]
        lora_wight = float(lora_id_a[1])
        de_name = lora_id_a[2]
        if lora_id.isdigit():
            logger.info(f'load this {lora_id}')
            response, output_path, filename, total_size = download_file_c(lora_id)
            CHUNK_SIZE = 1638400
            output_file = os.path.join(output_path, filename)
            if not os.path.exists(output_file):
                with open(output_file, 'wb') as f:
                    downloaded = 0
                    start_time = time.time()
                    logger.info(f'start do this {start_time}')
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
                            notify_it('ws', 'processing_progress', {'progress': progress}, to=room_id)
                            # sys.stdout.write(f'\rDownloading: {filename} [{progress * 100:.2f}%] - {speed:.2f} MB/s')
                            # sys.stdout.flush()
            else:
                logger.info(f'had down loaded this {lora_id} {output_file}')
        else:
            output_file = f"/mnt/fast/civitai-downloader/lora/{lora_id}"
        logger.info(f'loading this {lora_id} {lora_wight} {de_name} {output_file}')
        processor = sdxlInpainting()
        if lora_wight < 0:
            processor.unload_lora_weights()
            lora_wight = 0 - lora_wight
        processor.load_lora_weights(lora_path = output_file, adapter_name=de_name, lora_scale= lora_wight)
        logger.info(f'loaded this {lora_id} {lora_wight} {de_name} {output_file}')
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
    notify_it('ws', 'processing_done_text', {'text': res_text}, to=room_id)

@socketio.on('process_image_remve')
def process_image_remve(data):
    processor = sdxlInpainting()
    processor.unload_lora_weights()
def room_path(room_id):
    return os.path.join(app.config['UPLOAD_FOLDER'], room_id)

def progress_callback(step, t, latents, had, room_id, num_inference_steps, notify_type):
    progress = int((step / num_inference_steps) * 100)
    notify_it(notify_type, 'processing_progress', {'progress': progress, 'had': had}, to=room_id)
    if room_id in stop_it_set:
        stop_it_set.remove(room_id)
        notify_it(notify_type, 'processing_progress', {'progress': 'stop', 'had': had}, to=room_id)
        return True

# 定义回调函数并传递 'had' 参数
def create_callback(had_value, room_id, num_inference_steps, notify_type):
    return lambda step, t, latents: progress_callback(step, t, latents, had_value, room_id, num_inference_steps, notify_type)

@socketio.on('process_replace_image_b')
def process_replace_image_b(data):
    logger.info(f'get it process_replace_image_b')
    # 读取图像
    filename_face = data['filename_face']
    filename_handler = data['filename_handler']
    data['notify_type'] = 'ws'
    notify_type = data['notify_type']
    room_id = data['room_id']
    path = room_path(room_id)
    logger.info(f'get it process_replace_image_b {filename_face} {filename_handler} {room_id} {path}')
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
    logger.info(f'get it {re_fina}')
    re_fina_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, re_fina)
    logger.info(f'start save it {re_fina_path}')
    cv2.imwrite(re_fina_path, processed_img)
    logger.info(f'suc save it {re_fina}')
    room_image_manager.insert_imgStr(room_id, f'{re_fina}', 'done', 'book', notify_fuc=notify_it, notify_type=notify_type)
    notify_it(notify_type, 'processing_step_fin', {'fin': 'f'}, to=room_id)

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


@socketio.on('process_image_b')
def add_image_processing_task(data):
    data['notify_type'] = 'ws'
    logger.info(f'get req is {data}')
    data['def_skin'] = 'inpaint'
    roomId = data['roomId']
    room_image_manager = RoomImageManager()
    user_info = query_or_def(User(roomId, roomId))
    add_task_list(data, 'ws_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)

@socketio.on('process_text_gen_pic')
def process_text_gen_pic(data):
    roomId = data['roomId']
    data['roomId'] = roomId
    data['notify_type'] = 'ws'
    data['def_skin'] = '909'
    prompt = data['prompt']
    reverse_prompt = data['reverse_prompt']
    face_filename = data['face_filename']
    filename = data['filename']
    gen_type = data['gen_type'] # 'flux' 或 ''
    logger.info(f'get req is {data}')
    room_image_manager = RoomImageManager()
    user_info = query_or_def(User(roomId, roomId))
    add_task_list(data, 'ws_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)

@socketio.on('process_pic_find_face')
def process_pic_find_face(data):
    roomId = data['roomId']
    data['notify_type'] = 'ws'
    data['def_skin'] = 'swap_face'
    pre_face_pic_list = data['pre_face_pic_list']
    logger.info(f'get req is {data}')
    room_image_manager = RoomImageManager()
    user_info = query_or_def(User(roomId, roomId))
    add_task_list(data, 'ws_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)

@socketio.on('process_pic_swap_face')
def process_pic_swap_face(data):
    roomId = data['roomId']
    org_faces = data['org_faces']
    to_faces = data['to_faces']
    filename = data['filename']
    data['notify_type'] = 'ws'
    data['def_skin'] = 'start_swap_face'
    logger.info(f'get req is {data}')
    room_image_manager = RoomImageManager()
    user_info = query_or_def(User(roomId, roomId))
    add_task_list(data, 'ws_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)

@socketio.on('process_video_swap_face')
def process_video_swap_face(data):
    roomId = data['roomId']
    data['notify_type'] = 'ws'
    data['def_skin'] = 'start_swap_face_video'
    filename = data['filename']
    org_faces = data['org_faces']
    to_faces = data['to_faces']
    logger.info(f'get req is {data}')
    room_image_manager = RoomImageManager()
    user_info = query_or_def(User(roomId, roomId))
    add_task_list(data, 'ws_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)


# 同步更新 task_positions 中任务的队列位置

def update_task_positions(finished_room_id, task_positions):
    # 遍历字典，减小所有 room_id 的位置
    for room_id in task_positions:
        if task_positions[room_id] > task_positions.get(finished_room_id, 0):
            task_positions[room_id] -= 1
# 消费者线程，持续从队列中获取任务进行处理
def queue_consumer():
    while True:
        # 从队列中获取任务
        task_data = glob_task_queue.get()
        room_id = task_data.get('roomId')
        notify_type = task_data.get('notify_type')
        task_id = task_data.get('task_id')  # 获取任务的唯一 ID
        notify_fuc_name = task_data.get('notify_fuc')
        user_id = task_data.get('user_id')
        notify_fuc = function_mapping.get(notify_fuc_name)  # 通过映射字典获取函数
        if task_data is None:
            break
        try:
            logger.info(f'run it {task_data}')
            handle_image_processing_b(task_data, notify_fuc, app_path, room_image_manager, create_callback)
            user_info = query_or_def(User(user_id))
            user_vip_control(user_info, -1)
        except Exception as e:
            logger.info(f"task error: {e}")
            traceback.print_exc()
            logger.info(f"processing error is  -------: {e}")
            notify_fuc(notify_type, 'processing_step_progress', {'text': '抱歉book yes 异常了请重新操作'},
                       to=room_id,keyList=get_rest_key())
        finally:
            glob_task_queue.task_done()
            # 从数据库中删除已处理的任务，使用 task_id 精确删除
            room_image_manager.remove_task_by_id('glob_task_queue', task_id)
            # 任务已被消费，移除它的 room_id
            if room_id in glob_task_positions:
                del glob_task_positions[room_id]
                # 同步更新剩余任务的位置
            update_task_positions(room_id, glob_task_positions)

def queue_consumer_video():
    while True:
        # 从队列中获取任务
        task_video_data = task_vide_queue.get()
        room_id = task_video_data.get('roomId')
        notify_type = task_video_data.get('notify_type')
        task_id = task_video_data.get('task_id')  # 获取任务的唯一 ID
        notify_fuc_name = task_video_data.get('notify_fuc')
        user_id = task_video_data.get('user_id')
        notify_fuc = function_mapping.get(notify_fuc_name)  # 通过映射字典获取函数
        if task_video_data is None:
            break
        try:
            handle_image_processing_b(task_video_data, notify_fuc, app_path, room_image_manager, create_callback)
            user_info = query_or_def(User(user_id))
            user_vip_control(user_info, -1)
        except Exception as e:
            logger.info(f"task error: {e}")
            traceback.print_exc()
            notify_fuc(notify_type, 'processing_step_progress', {'text': '抱歉book yes 异常了请重新操作'},
                       to=room_id, keyList=get_rest_key())
        finally:
            task_vide_queue.task_done()
            # 从数据库中删除已处理的任务，使用 task_id 精确删除
            room_image_manager.remove_task_by_id('task_vide_queue', task_id)
            # 任务已被消费，移除它的 room_id
            if room_id in video_task_positions:
                del video_task_positions[room_id]
                # 同步更新剩余任务的位置
            update_task_positions(room_id, video_task_positions)

# 启动消费者线程
consumer_thread = threading.Thread(target=queue_consumer, daemon=True)
consumer_thread.start()

# 启动消费者线程
consumer_video_thread = threading.Thread(target=queue_consumer_video, daemon=True)
consumer_video_thread.start()

# 启动 Flask 和 socket.io
def run_socketio():
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)


if __name__ == '__main__':
    room_image_manager = RoomImageManager()
    # room_image_manager.clear_tables()
    # 恢复 glob_task_queue 中未完成的任务
    load_unfinished_tasks('glob_task_queue', glob_task_queue, room_image_manager)
    # 恢复 task_vide_queue 中未完成的任务（如果有多个队列需要恢复）
    load_unfinished_tasks('task_vide_queue', task_vide_queue, room_image_manager)
    # 将 Telegram bot 放在主线程，socketio 放在子线程
    telegram_thread = threading.Thread(target=run_socketio)
    # 启动 socketio 线程
    telegram_thread.start()
    # 获取主线程的事件循环
    loop = asyncio.get_event_loop()
    # 启动队列消费者线程
    consumer_thread = threading.Thread(target=queue_tel_consumer, args=(loop,), daemon=True)
    consumer_thread.start()

    # 运行 Telegram bot 在主线程
    run_telegram_bot()

    # 等待线程完成
    telegram_thread.join()
