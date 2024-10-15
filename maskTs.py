import os
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip
from PIL import Image
from mask_generator import MaskGenerator
import cv2
import numpy as np
import gc

def extract_audio_and_frames(video_path, frames_dir, audio_path):
    # 创建帧图片的输出目录
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    # 打开视频文件
    video_clip = VideoFileClip(video_path)

    # 保存音频
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)

    # 保存每一帧的图片
    for i, frame in enumerate(video_clip.iter_frames()):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        Image.fromarray(frame).save(frame_path)

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


def handlerPic(mask_generator, input_path, output_path, prev_image_path=None, prev_output_path=None):
    print(f"start first pic h ... {input_path} to {output_path}")
    init_image = Image.open(input_path).convert("RGB")
    prev_image = None
    flow = None

    if prev_image_path and prev_output_path:
        prev_image = Image.open(prev_image_path).convert("RGB")
        prev_image_np = np.array(prev_image)
        init_image_np = np.array(init_image)
        flow = compute_optical_flow(prev_image_np, init_image_np)

    mask_image_np = mask_generator.generate_mask_f(init_image)

    if flow is not None:
        mask_image = apply_flow_to_mask(mask_image_np, flow)
        mask_image = Image.fromarray(mask_image)
    else:
        mask_image = Image.fromarray(mask_image_np, mode="L")
    mask_image.save(output_path)
    gc.collect()


def reassemble_video(frames_dir, audio_path, output_video_path, mask_generator=None):
    print("start reassemble video ...")
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])

    processed_frames_dir = os.path.join(frames_dir, "processed")
    if not os.path.exists(processed_frames_dir):
        os.makedirs(processed_frames_dir)

    total_frames = len(frame_files)
    processed_frame_files = []

    for i, frame_file in enumerate(frame_files):
        # prev_frame_file = frame_files[i - 1] if i > 0 else None
        # prev_output_frame_file = processed_frame_files[i - 1] if i > 0 else None
        output_frame_path = os.path.join(processed_frames_dir, os.path.basename(frame_file))
        # handlerPic(mask_generator, frame_file, output_frame_path, prev_frame_file, prev_output_frame_file)
        processed_frame_files.append(output_frame_path)
        print(f"Processed {i + 1}/{total_frames} frames.")

    processed_clips = [ImageClip(img_path).set_duration(1 / 24) for img_path in processed_frame_files]
    video_clip = concatenate_videoclips(processed_clips, method="compose")

    original_video = VideoFileClip(video_path)
    fps = original_video.fps
    original_video.close()

    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)

    final_clip.write_videofile(output_video_path, codec="libx264", fps=fps, audio_codec="aac")

    video_clip.close()
    audio_clip.close()
    final_clip.close()


if __name__ == "__main__":
    video_path = "480p.mp4"
    frames_dir = "frames-480p"
    audio_path = "480p.mp3"
    output_video_path = "480p_output_video.mp4"
    mask_generator = MaskGenerator()
    reassemble_video(frames_dir, audio_path, output_video_path, mask_generator)