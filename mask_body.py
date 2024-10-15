import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image, ImageFilter


# 定义函数以生成不包括头部的人体掩码
def mark_human_body_without_head(image_path):
    # 加载图像处理器和模型
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")

    # 加载图像并进行预处理
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # 获取分割结果
    prediction = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    segmentation = prediction['segmentation']
    segments_info = prediction['segments_info']

    # 获取标签信息
    person_segments = [segment for segment in segments_info if segment['label_id'] == 0]  # "person" 标签是0

    # 创建掩码
    mask = Image.new("L", image.size, 0)
    for segment in person_segments:
        segment_id = segment['id']
        for y in range(segmentation.shape[0]):
            for x in range(segmentation.shape[1]):
                if segmentation[y, x] == segment_id:
                    mask.putpixel((x, y), 255)

    # 对掩码进行模糊处理，使边界更平滑
    mask = mask.filter(ImageFilter.GaussianBlur(5))
    return mask


# 示例使用
image_path = "cb45ac85254a115f83761faff0c4e150.png"
mask = mark_human_body_without_head(image_path)
mask.show()