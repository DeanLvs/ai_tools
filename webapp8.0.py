import os
import torch
from downloadC import download_file_c
from CoreService import resize_image, generate_unique_key, re_auto_mask_with_out_head_b, convert_to_ndarray
import time
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from PIL import ImageDraw,Image, ImageFilter
from ImageProcessorSDXAllCN import CustomInpaintPipeline
import base64
import numpy as np
import requests
import gc
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
socketio = SocketIO(app)
# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Dictionary to store processed results
processed_results = {}

processed_user_results={}
stop_it_set={}

import shutil
import cv2

glob_parts_to_include = None
@app.route('/')
def index():
    return render_template('index.html')


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

        # 备份原始图像
        backup_path = os.path.join(app.config['UPLOAD_FOLDER'], f"backup_{filename}")
        shutil.copy(file_path, backup_path)

        # 调整图像大小
        resize_image(file_path)
        return jsonify({'filename': filename, 'file_url': f'/uploads/{filename}'})

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

def genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, genImage, big_mask,
          num_inference_steps, guidance_scale, progress_callback, strength=0.8,
          control_image=[None, None, None, None, None], controlnet_conditioning_scale=0.7, seed=123):
    try:
        print(f"gen use prompt: {prompt} reverse_prompt: {reverse_prompt} "
              f"num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}, "
              f"strength: {strength} prompt_2: {prompt_2} reverse_prompt_2: {reverse_prompt_2} "
              f"control_image: {control_image} controlnet_conditioning_scale_t: {controlnet_conditioning_scale}"
              f"seed: {seed}")
        # 0 -- openpose
        # 1 -- depth
        # 2 -- hed/pidi/scribble/ted
        # 3 -- canny/lineart/anime_lineart/mlsd
        # 4 -- normal
        # 5 -- segment
        # image_list must be compatible with union_control_type
        with torch.no_grad():
            control_image_array = convert_to_ndarray(control_image[0])
            img_seg_array = convert_to_ndarray(control_image[2])
            height, width, _ = control_image_array.shape

            ratio = np.sqrt(1024. * 1024. / (width * height))
            new_width, new_height = int(width * ratio), int(height * ratio)

            controlnet_img_pose = cv2.resize(control_image_array, (new_width, new_height))
            controlnet_img_seg = cv2.resize(img_seg_array, (new_width, new_height))
            controlnet_img_pose = Image.fromarray(controlnet_img_pose)
            controlnet_img_seg = Image.fromarray(controlnet_img_seg)

            generator = torch.Generator('cuda').manual_seed(seed)
            resultTemp = processor.pipe(
                          prompt=[prompt] * 1,
                          image=genImage,
                          mask_image=big_mask,
                          control_image_list=[controlnet_img_pose, 0, 0, 0, 0, controlnet_img_seg],
                          negative_prompt=[reverse_prompt] * 1,
                          generator=generator,
                          width=new_width,
                          height=new_height,
                          num_inference_steps=30,
                          union_control=True,
                          union_control_type=torch.Tensor([1, 0, 0, 0, 0, 1]),
                          crops_coords_top_left=(0, 0),
                          target_size=(new_width, new_height),
                          original_size=(new_width * 2, new_height * 2),
                          callback=progress_callback,
                          callback_steps=1
                         ).images[0]

    finally:
        # 清理已使用的张量
        # 在使用完后显式删除变量以释放内存
        # 释放 GPU 内存
        torch.cuda.empty_cache()
        gc.collect()
    return resultTemp
@socketio.on('process_set_lora')
def process_set_lora(data):
    print("got it process_set_lora")
    room_id = data['roomId']
    if 'lora_id' in data and data['lora_id'] == 'x':
        processor = CustomInpaintPipeline()
        processor.release_resources()
        emit('processing_done_text', {'text': 'xxx'}, broadcast=True, to=room_id)
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
            output_file = f"/mnt/sessd/civitai-downloader/lora/{lora_id}"
        print(f'loading this {lora_id} {lora_wight} {de_name} {output_file}')
        processor = CustomInpaintPipeline()
        if lora_wight < 0:
            processor.unload_lora_weights()
            lora_wight = 0 - lora_wight
        processor.load_lora_weights(lora_path = output_file, adapter_name=de_name, lora_scale= lora_wight)
        print(f'loaded this {lora_id} {lora_wight} {de_name} {output_file}')
        res_text = '完成' + output_file
    elif 'wei_id' in data and data['wei_id'] != '':
        wei_id = data['wei_id']
        processor = CustomInpaintPipeline()
        wei_id_a = wei_id.split('#')
        load_part_list = []
        weights_path = f"/mnt/sessd/civitai-downloader/{wei_id_a[0]}"
        for par in wei_id_a:
            load_part_list.append(par)
        # load_part_list = ['unet', 'vae', 't1', 't2']
        processor.load_pretrained_weights(weights_path, load_part_list)
        res_text = '完成'+weights_path
    emit('processing_done_text', {'text': res_text}, broadcast=True, to=room_id)



def save_all(img_to_save, img_to_save_path, save_file_name):
    img_to_save.save(img_to_save_path)
    return save_file_name
@socketio.on('process_image_remve')
def process_image_remve(data):
    processor = CustomInpaintPipeline()
    processor.unload_lora_weights()

@socketio.on('process_image_b')
def handle_image_processing_b(data):
    print("got it process_image_b")
    def_skin = str(data['def_skin'])
    filename = data['filename']
    prompt = data['prompt']
    prompt_2 = data['prompt_2']
    re_mask = data['re_mask']
    re_p_float_array = eval(data['re_p'])
    re_b_float_array = eval(data['re_b'])
    ha_p = int(data['ha_p'])
    ga_b = float(data['ga_b'])
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
        print(f'{def_skin}')
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
    unique_key = generate_unique_key(filename, prompt, reverse_prompt, num_inference_steps, guidance_scale, seed, re_p_float_array, re_b_float_array, ha_p, ga_b, strength,prompt_2, reverse_prompt_2)
    print(f"unique_key: {unique_key} processed_results: {processed_results} ")
    processed_r_r_filename = f'processed_r_r_{unique_key}.png'
    processed_r_filename = f'processed_r_{unique_key}.png'
    clear_f_fname = f'processed_clear_r_{unique_key}_' + filename
    processed_org_r_filename = f'processed_org_r_{unique_key}.png'
    fill_all_processed_org_r_filename = f'f_processed_org_r_{unique_key}.png'
    if unique_key in processed_results:
        processed_filename = processed_results[unique_key]
        # emit('processing_done', {'processed_image_url': f'/uploads/{processed_filename}'}, broadcast=True, to=room_id)
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_r_r_filename}'}, broadcast=True, to=room_id)
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_r_filename}'}, broadcast=True, to=room_id)
        emit('processing_done', {'processed_image_url': f'/uploads/{clear_f_fname}'}, broadcast=True, to=room_id)
        emit('processing_done', {'processed_image_url': f'/uploads/{processed_org_r_filename}'}, broadcast=True, to=room_id)
        return
    else:
        processed_results[unique_key] = unique_key
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mask_clear_finally = None
    control_image_return = None
    control_image = None
    print(f"re_mask: {re_mask} ")

    filled_image_pil_return = None
    clear_image_pil_return = None
    next_filled_image_pil_return = None
    filled_image_pil_filename = 'filled_image_pil_' + filename
    filled_image_pil_path = os.path.join(app.config['UPLOAD_FOLDER'], filled_image_pil_filename)

    next_filled_image_pil_filename = 'filled_image_pil_next_' + filename
    next_filled_image_pil_path = os.path.join(app.config['UPLOAD_FOLDER'], next_filled_image_pil_filename)

    cur_res_filename = 'clear_return_' + filename
    cur_res_path = os.path.join(app.config['UPLOAD_FOLDER'], cur_res_filename)

    mask_clear_finally_filename = 'filled_use_mask_image_pil_' + filename
    mask_clear_finally_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_clear_finally_filename)

    segmentation_image_pil_filename = 'seg_use_image_pil_' + filename
    segmentation_image_pil_path = os.path.join(app.config['UPLOAD_FOLDER'], segmentation_image_pil_filename)

    control_net_fname = f'control_net_' + filename
    control_net_fname_path = os.path.join(app.config['UPLOAD_FOLDER'], control_net_fname)

    nor_control_net_fname = f'nor_control_net_' + filename
    nor_control_net_fname_path = os.path.join(app.config['UPLOAD_FOLDER'], nor_control_net_fname)

    cor_control_net_fname = f'cor_control_net_' + filename
    cor_control_net_fname_path = os.path.join(app.config['UPLOAD_FOLDER'], cor_control_net_fname)

    fill_all_mask_img_fname = f'fill_all_skin_' + filename
    fill_all_mask_img_name_path = os.path.join(app.config['UPLOAD_FOLDER'], fill_all_mask_img_fname)

    mask_future = None
    gen_fix_pic_future = None
    normal_map_img = None
    normal_map_img_return = None
    color_image_r = None
    fill_all_mask_future = None
    segmentation_image_pil = None
    fill_all_mask_img = None
    if re_mask != 'F' or not os.path.exists(filled_image_pil_path):
        print(f"filled_image_pil_filename: {filled_image_pil_filename} ")
        emit('processing_step_progress', {'text': 'book yes for 不可言说中...'}, broadcast=True, to=room_id)
        mask_future, gen_fix_pic_future, fill_all_mask_future, segmentation_image_pil = re_auto_mask_with_out_head_b(app.config['UPLOAD_FOLDER'], filename)

    image_org = Image.open(file_path)
    original_width, original_height = image_org.size
    orgImage = image_org.convert("RGB")

    if segmentation_image_pil:
        save_all(segmentation_image_pil, segmentation_image_pil_path, segmentation_image_pil_filename)
    elif os.path.exists(segmentation_image_pil_path):
        segmentation_image_pil = Image.open(segmentation_image_pil_path).convert("RGB")
    def progress_callback(step, t, latents, had):
        progress = int((step / num_inference_steps) * 100)
        emit('processing_progress', {'progress': progress, 'had': had}, broadcast=True, to=room_id)
        if room_id in stop_it_set:
            stop_it_set.remove(room_id)
            emit('processing_progress', {'progress': 'stop', 'had': had}, broadcast=True, to=room_id)
            return True

    # 定义回调函数并传递 'had' 参数
    def create_callback(had_value):
        return lambda step, t, latents: progress_callback(step, t, latents, had_value)

    try:
        if mask_future is not None:
            mask_clear_finally, temp_humanMask, control_image_return, normal_map_img_return, color_image_r = mask_future.result()
            emit('processing_step_progress', {'text': '识别图像中...'}, broadcast=True, to=room_id)
            save_all(mask_clear_finally, mask_clear_finally_path, mask_clear_finally_filename)
            emit('processing_done', {'processed_image_url': f'/uploads/{mask_clear_finally_filename}','keephide':'k'}, broadcast=True, to=room_id)
            imgStrList.append(f'/uploads/{mask_clear_finally_filename}')

            save_all(control_image_return, control_net_fname_path, control_net_fname)
            if normal_map_img_return:
                save_all(normal_map_img_return, nor_control_net_fname_path, nor_control_net_fname)

            if color_image_r:
                save_all(color_image_r, cor_control_net_fname_path, cor_control_net_fname)

            if ha_p > 0:
                emit('processing_done', {'processed_image_url': f'/uploads/{control_net_fname}', 'keephide': 'k'},
                 broadcast=True, to=room_id)
                imgStrList.append(f'/uploads/{control_net_fname}')
                if normal_map_img_return:
                    emit('processing_done', {'processed_image_url': f'/uploads/{nor_control_net_fname}', 'keephide': 'k'},
                     broadcast=True, to=room_id)
                    imgStrList.append(f'/uploads/{nor_control_net_fname}')
                if color_image_r:
                    emit('processing_done',
                         {'processed_image_url': f'/uploads/{cor_control_net_fname}', 'keephide': 'k'},
                         broadcast=True, to=room_id)
                    imgStrList.append(f'/uploads/{cor_control_net_fname}')

        emit('processing_step_progress', {'text': 'book yes 加载模型中...'}, broadcast=True, to=room_id)
        processor = CustomInpaintPipeline()
        emit('processing_step_progress', {'text': 'book yes 加载模型完成...'}, broadcast=True, to=room_id)

        if isinstance(mask_clear_finally, Image.Image) and mask_clear_finally is not None:
            big_mask = mask_clear_finally
        elif isinstance(mask_clear_finally_path, str) and os.path.exists(mask_clear_finally_path):
            big_mask = Image.open(mask_clear_finally_path).convert("L")

        if isinstance(control_image_return, Image.Image) and control_image_return is not None:
            print(f'control_image_return is {type(control_image_return)}')
            control_image = control_image_return
            print(f'control_image_return is {type(control_image)}')
        elif isinstance(control_net_fname_path, str) and os.path.exists(control_net_fname_path):
            print(f'open {control_net_fname_path} use it')
            control_image = Image.open(control_net_fname_path).convert("RGB")
            print(f'control_image is {type(control_image)}')

        if isinstance(normal_map_img_return, Image.Image) and normal_map_img_return is not None:
            normal_map_img = normal_map_img_return
        elif isinstance(nor_control_net_fname_path, str) and os.path.exists(nor_control_net_fname_path):
            normal_map_img =Image.open(nor_control_net_fname_path).convert("RGB")

        if isinstance(color_image_r, Image.Image) and color_image_r is not None:
            cor_map_img = color_image_r
        elif isinstance(cor_control_net_fname_path, str) and os.path.exists(cor_control_net_fname_path):
            cor_map_img =Image.open(cor_control_net_fname_path).convert("RGB")

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = None,None,None,None

        if fill_all_mask_future is not None:
            fill_all_mask_img = fill_all_mask_future.result()
            if fill_all_mask_img is not None:
                save_all(fill_all_mask_img, fill_all_mask_img_name_path, fill_all_mask_img_fname)

                emit('processing_done', {'processed_image_url': f'/uploads/{fill_all_mask_img_fname}', 'keephide': 'k'},
                     broadcast=True, to=room_id)
                imgStrList.append(f'/uploads/{fill_all_mask_img_fname}')
        elif os.path.exists(fill_all_mask_img_name_path):
            fill_all_mask_img = Image.open(filled_image_pil_path).convert("RGB")

        def genItOrg():
            emit('processing_step_progress', {'text': 'book yes 开始生成4张...'}, broadcast=True, to=room_id)
            orgResult = genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, orgImage, big_mask, num_inference_steps, guidance_scale,
                              create_callback('1/4'), strength, [control_image, normal_map_img, segmentation_image_pil, cor_map_img])
            orgResult = orgResult.resize((original_width, original_height), Image.LANCZOS)

            processed_org_r_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_org_r_filename)
            orgResult.save(processed_org_r_file_path)
            emit('processing_done', {'processed_image_url': f'/uploads/{processed_org_r_filename}'}, broadcast=True,
                 to=room_id)
            imgStrList.append(f'/uploads/{processed_org_r_filename}')
            del orgResult

        genItOrg()

        if fill_all_mask_img:
            def genItFillAll():
                fillOrgResult = genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, fill_all_mask_img, big_mask,
                                  num_inference_steps, guidance_scale,
                                  create_callback('2/4'), strength,
                                      [control_image, normal_map_img, segmentation_image_pil, cor_map_img])
                fillOrgResult = fillOrgResult.resize((original_width, original_height), Image.LANCZOS)

                f_processed_org_r_file_path = os.path.join(app.config['UPLOAD_FOLDER'], fill_all_processed_org_r_filename)
                fillOrgResult.save(f_processed_org_r_file_path)
                emit('processing_done', {'processed_image_url': f'/uploads/{fill_all_processed_org_r_filename}'}, broadcast=True,
                     to=room_id)
                imgStrList.append(f'/uploads/{fill_all_processed_org_r_filename}')
                del fillOrgResult

            genItFillAll()

        if gen_fix_pic_future is not None:
            clear_image_pil_return, filled_image_pil_return, next_filled_image_pil_return = gen_fix_pic_future.result()
            save_all(clear_image_pil_return, cur_res_path, cur_res_filename)
            emit('processing_done', {'processed_image_url': f'/uploads/{cur_res_filename}'}, broadcast=True, to=room_id)
            imgStrList.append(f'/uploads/{cur_res_filename}')
            if filled_image_pil_return:
                save_all(filled_image_pil_return, filled_image_pil_path, filled_image_pil_filename)
                emit('processing_done', {'processed_image_url': f'/uploads/{filled_image_pil_filename}'}, broadcast=True, to=room_id)
                imgStrList.append(f'/uploads/{filled_image_pil_filename}')

            save_all(next_filled_image_pil_return, next_filled_image_pil_path, next_filled_image_pil_filename)
            emit('processing_done', {'processed_image_url': f'/uploads/{next_filled_image_pil_filename}'}, broadcast=True,
                 to=room_id)
            imgStrList.append(f'/uploads/{next_filled_image_pil_filename}')

        if isinstance(clear_image_pil_return, Image.Image) and clear_image_pil_return is not None:
            clear_org_i = clear_image_pil_return
        elif isinstance(cur_res_path, str) and os.path.exists(cur_res_path):
            clear_org_i = Image.open(cur_res_path).convert("RGB")
        def genitOrgFix():
            org_fix_Result = genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2, clear_org_i, big_mask, num_inference_steps, guidance_scale,
                              create_callback('2/4'), strength, [control_image, normal_map_img, segmentation_image_pil, cor_map_img])
            org_fix_Result = org_fix_Result.resize((original_width, original_height), Image.LANCZOS)

            org_fix_Result.save(os.path.join(app.config['UPLOAD_FOLDER'], clear_f_fname))
            emit('processing_done', {'processed_image_url': f'/uploads/{clear_f_fname}'}, broadcast=True,
                 to=room_id)
            imgStrList.append(f'/uploads/{clear_f_fname}')

            del org_fix_Result

        genitOrgFix()

        genImage = None
        if isinstance(filled_image_pil_return, Image.Image) and filled_image_pil_return is not None:
            genImage = filled_image_pil_return
        elif isinstance(filled_image_pil_path, str) and os.path.exists(filled_image_pil_path):
            genImage = Image.open(filled_image_pil_path).convert("RGB")
        if isinstance(genImage, Image.Image) and genImage is not None:
            def genItFilledHuman():
                result = genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2,
                               genImage, big_mask, num_inference_steps, guidance_scale,
                               create_callback('3/4'), strength, [control_image, normal_map_img, segmentation_image_pil, cor_map_img])
                result = result.resize((original_width, original_height), Image.LANCZOS)

                processed_r_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_r_filename)
                result.save(processed_r_file_path)
                emit('processing_done', {'processed_image_url': f'/uploads/{processed_r_filename}'}, broadcast=True, to=room_id)
                imgStrList.append(f'/uploads/{processed_r_filename}')
                emit('processing_step_fin', {'fin': 'f'}, broadcast=True, to=room_id)

            genItFilledHuman()

        next_genImage = None
        if next_filled_image_pil_return:
            next_genImage = next_filled_image_pil_return
        elif isinstance(next_filled_image_pil_path, str) and os.path.exists(next_filled_image_pil_path):
            next_genImage = Image.open(next_filled_image_pil_path).convert("RGB")
        if next_genImage:
            result_next = genIt(processor, prompt, prompt_2, reverse_prompt, reverse_prompt_2,
                                next_genImage, big_mask, num_inference_steps, guidance_scale,
                                create_callback('4/4'), strength, [control_image, normal_map_img, segmentation_image_pil, cor_map_img])
            result_next = result_next.resize((original_width, original_height), Image.LANCZOS)

            processed_r_r_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_r_r_filename)
            result_next.save(processed_r_r_file_path)
            emit('processing_done', {'processed_image_url': f'/uploads/{processed_r_r_filename}'}, broadcast=True, to=room_id)
            imgStrList.append(f'/uploads/{processed_r_r_filename}')
            emit('processing_step_fin', {'fin': 'f'}, broadcast=True, to=room_id)
            del result_next
    finally:
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
    path = app.config['UPLOAD_FOLDER']
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
    re_fina_path = os.path.join(app.config['UPLOAD_FOLDER'], re_fina)
    print(f'start save it {re_fina_path}')
    cv2.imwrite(re_fina_path, processed_img)
    print(f'suc save it {re_fina}')
    imgStrList = []
    if room_id:
        if room_id in processed_user_results:
            imgStrList = processed_user_results[room_id]
        else:
            processed_user_results[room_id] = imgStrList
    emit('processing_done', {'processed_image_url': f'/uploads/{re_fina}'}, broadcast=True, to=room_id)
    imgStrList.append(f'/uploads/{re_fina}')
    emit('processing_step_fin', {'fin': 'f'}, broadcast=True, to=room_id)

@app.route('/rep_upload_face', methods=['POST'])
def rep_upload_face():
    if 'file_face_img' not in request.files:
        return 'No file part'
    file_face_img = request.files['file_face_img']
    if file_face_img.filename == '':
        return 'No selected file'
    if file_face_img:
        file_face_img_filename = 'r_f_'+file_face_img.filename
        file_face_img_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_face_img_filename)
        file_face_img.save(file_face_img_file_path)

        # 调整图像大小
        resize_image(file_face_img_file_path)
        return jsonify({'filename_face': file_face_img_filename, 'file_face_url': f'/uploads/{file_face_img_filename}'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, allow_unsafe_werkzeug=True)