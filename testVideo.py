class video_pre_info:
    def __init__(self, index):
        self.org_name = f"org_{index:04d}.png"
        self.mask_name = f"mask_{index:04d}.png"
        self.pose_name = f"pose_{index:04d}.png"
        self.depth_name = f"depth_{index:04d}.png"
        self.depth_3d_name = f"depth_3d_{index:04d}.png"
        self.line_name = f"line_{index:04d}.png"
        self.clear_name = f"clear_{index:04d}.png"
        self.skin_name = f"skin_{index:04d}.png"

def handle_image_processing_video(room_id, video_file_name):
    temp_path = video_file_name.repalce('.mp4', '')
    temp_path = os.path.join('/mnt/fast/mp4_temp', room_id, temp_path)
    os.makedirs(temp_path, exist_ok=True)


    room_upload_folder = room_path(room_id)
    video_path = os.path.join(room_upload_folder, video_file_name)

    # 打开视频文件
    video_clip = VideoFileClip(video_path)
    processed_frames_dir = os.path.join(temp_path, "processed")
    # 保存音频
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(temp_path + "/maninit.mp3")

    # 保存每一帧的图片并立即处理
    for i, frame in enumerate(video_clip.iter_frames()):
        v_p_info = video_pre_info(i)
        logger.info(f'start handler {i} {v_p_info}')
        frame_path = os.path.join(temp_path, v_p_info.org_name)
        output_frame_path = os.path.join(processed_frames_dir, v_p_info.org_name)
        if os.path.exists(output_frame_path):
            logger.info(f'Frame {i:04d} already processed.')
            continue
        Image.fromarray(frame).save(frame_path)
        mask_future, gen_fix_pic_future, fill_all_mask_future = re_auto_mask_with_out_head_b(temp_path, v_p_info.org_name, False)
        mask_clear_finally, temp_humanMask, control_image_return, normal_map_img_return, normal_3d_map_img_return, with_torso_mask, f_output_line_image_pil, segmentation_image_pil = mask_future.result()
        # 重绘区域
        with_torso_mask.save(os.path.join(temp_path, v_p_info.mask_name))
        # 姿势识别
        control_image_return.save(os.path.join(temp_path, v_p_info.pose_name))
        # 深度识别
        normal_map_img_return.save(os.path.join(temp_path, v_p_info.depth_name))
        # 3d深度识别
        normal_3d_map_img_return.save(os.path.join(temp_path, v_p_info.depth_3d_name))
        # 排除头部手臂的边线
        f_output_line_image_pil.save(os.path.join(temp_path, v_p_info.line_name))

        clear_image_pil_return, filled_image_pil_return, next_filled_image_pil_return = gen_fix_pic_future.result()
        # 清除后图片
        clear_image_pil_return.save(os.path.join(temp_path, v_p_info.clear_name))
        # 自动识别填充
        filled_image_pil_return.save(os.path.join(temp_path, v_p_info.skin_name))
        logger.info(f'finish handler {i}')
    video_clip.close()
    audio_clip.close()

    # 释放内存
    del video_clip
    del audio_clip
    gc.collect()
    logger.info("Memory cleared after extracting audio and frames.")