import os
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip
from PIL import Image
# from ImageProcessorSDX import CustomInpaintPipeline
# from mask_generator import MaskGenerator
# from HandlerVideoLa import handle_image_processing_b
import cv2
import numpy as np
# import torch
import gc
import time


def extract_audio_and_frames(video_path, frames_dir, audio_path):
    # 创建帧图片的输出目录
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    processed_frames_dir = os.path.join(frames_dir, "processed")
    processed_frames_next_dir = os.path.join(frames_dir, "processednext")
    if not os.path.exists(processed_frames_dir):
        os.makedirs(processed_frames_dir)
    if not os.path.exists(processed_frames_next_dir):
        os.makedirs(processed_frames_next_dir)
    # 打开视频文件
    video_clip = VideoFileClip(video_path)

    # 保存音频
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)

    # 保存每一帧的图片并立即处理
    for i, frame in enumerate(video_clip.iter_frames()):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        output_frame_path = os.path.join(processed_frames_dir, f"frame_{i:04d}.png")
        other_output_frame_path = os.path.join(processed_frames_next_dir, f"frame_{i:04d}.png")
        if os.path.exists(output_frame_path) and os.path.exists(other_output_frame_path):
            print(f'Frame {i:04d} already processed.')
            continue
        Image.fromarray(frame).save(frame_path)
        handlerPic(frame_path, output_frame_path, other_output_frame_path)
    video_clip.close()
    audio_clip.close()

    # 释放内存
    del video_clip
    del audio_clip
    gc.collect()
    print("Memory cleared after extracting audio and frames.")


def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def apply_flow_to_mask(mask, flow):
    h, w = flow.shape[:2]
    # flow_map 的 x 和 y 方向可能需要调整
    flow_map_x = flow[..., 0]
    flow_map_y = flow[..., 1]

    # 生成每个像素的位置坐标
    coords_x, coords_y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    # 计算新的位置坐标，调整光流方向
    new_coords_x = coords_x + flow_map_x
    new_coords_y = coords_y + flow_map_y

    # 使用 np.clip 限制坐标范围在有效图像范围内，避免越界
    new_coords_x = np.clip(new_coords_x, 0, w - 1)
    new_coords_y = np.clip(new_coords_y, 0, h - 1)

    # 使用 cv2.remap 将 mask 映射到新的坐标
    remapped_mask = cv2.remap(mask, new_coords_x.astype(np.float32), new_coords_y.astype(np.float32), cv2.INTER_LINEAR)
    return remapped_mask


def handlerPic(input_path, output_path, other_output_frame_path):
    print(f"start processing frame {input_path} to {output_path} and {other_output_frame_path}")

    if os.path.exists(output_path) and os.path.exists(other_output_frame_path):
        print(f'Already exists: {output_path} {other_output_frame_path}')
        return

    try:
        result_image, fix_result_image = handle_image_processing_b(input_path)
        result_image.save(output_path)
        fix_result_image.save(other_output_frame_path)
        if 'fix_result_image' in locals():
            del fix_result_image
        if 'result_image' in locals():
            del result_image
        else:
            print("result_image does not exist.")
    except Exception as e:
        print(f"Error processing frame {input_path}: {e}")
        print("Saving original frame as the processed frame.")
        Image.open(input_path).save(output_path)
        Image.open(input_path).save(other_output_frame_path)

    torch.cuda.empty_cache()
    gc.collect()


def reassemble_video(frames_dir, audio_path, output_video_path, prc='processed', batch_size=100):
    print("start reassemble video ...")
    processed_frames_dir = os.path.join(frames_dir, prc)
    frame_files = sorted(
        [os.path.join(processed_frames_dir, f) for f in os.listdir(processed_frames_dir) if f.endswith('.png')])
    print(f"Number of frames found: {len(frame_files)}")
    if len(frame_files) == 0:
        print("No frames found. Exiting.")
        return
    try:
        processed_clips = []
        for i in range(0, len(frame_files), batch_size):
            batch_files = frame_files[i:i + batch_size]
            processed_clips.extend([ImageClip(img_path).set_duration(1 / 24) for img_path in batch_files])
            print(f"Processed batch {i // batch_size + 1}/{len(frame_files) // batch_size + 1}")
            time.sleep(3)  # 暂停几秒以缓解磁盘I/O压力

        video_clip = concatenate_videoclips(processed_clips, method="compose")

        original_video = VideoFileClip(video_path)
        fps = original_video.fps
        original_video.close()

        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        print("Writing video file...")
        final_clip.write_videofile(output_video_path, codec="libx264", fps=fps, audio_codec="aac")
        print("Video file created successfully.")
        video_clip.close()
        audio_clip.close()
        final_clip.close()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    video_path = "maninit.mp4"
    frames_dir = "frames-maninit"
    audio_path = "maninit.mp3"
    output_video_path = "maninit_output_video.mp4"
    output_video_path_next = "next_maninit_output_video.mp4"

    # extract_audio_and_frames(video_path, frames_dir, audio_path)
    reassemble_video(frames_dir, audio_path, output_video_path, prc='processed')

    reassemble_video(frames_dir, audio_path, output_video_path_next, prc='processednext')