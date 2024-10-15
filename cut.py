from moviepy.editor import VideoFileClip

# 输入视频文件路径
video_path = "input_video.mp4"
# 输出截取片段的文件路径
output_path = "output_clip.mp4"

# 定义截取的开始时间和结束时间
start_time = 2 * 60 + 50  # 2分50秒
end_time = None  # 直到视频结束

# 读取视频文件
video = VideoFileClip(video_path)

# 截取指定时间段
if end_time:
    video_clip = video.subclip(start_time, end_time)
else:
    video_clip = video.subclip(start_time)

# 保存截取的视频片段，保持原始分辨率
video_clip.write_videofile(output_path, codec="libx264", fps=video.fps, preset="medium", threads=4, audio_codec="aac")

# 释放资源
video.close()
video_clip.close()
