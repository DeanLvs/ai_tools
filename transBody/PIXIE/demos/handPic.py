import cv2

def print_image_dimensions(image_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 获取图片的高度和宽度
    height, width = image.shape[:2]

    # 输出图片的宽和高
    print(f"Image Width: {width}, Image Height: {height} , bi{width/height}")

# 示例用法

def crop_black_borders(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 找到所有非黑色像素的坐标
    coords = cv2.findNonZero(cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1])

    # 计算边界框
    x, y, w, h = cv2.boundingRect(coords)

    # 裁剪图像
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# 示例用法
# image = cv2.imread('IMG_5047_processed.jpg')  # 读取图像
# cropped_image = crop_black_borders(image)  # 裁剪四周黑色部分
# cv2.imwrite('cropped_image.jpg', cropped_image)  # 保存裁剪后的图像
# image_path = 'cropped_image.jpg'
# print_image_dimensions(image_path)
# image_path = 'IMG_5047.png'
# print_image_dimensions(image_path)

import cv2
import numpy as np


def resize_and_pad_image(image, target_size=1024):
    # 获取原始图像的宽和高
    original_height, original_width = image.shape[:2]

    # 计算缩放比例，保持宽高比，确保较长边等于目标尺寸
    scale = min(target_size / original_width, target_size / original_height)

    # 计算新的尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 计算填充的边界，保持图像在中心
    top_padding = (target_size - new_height) // 2
    bottom_padding = target_size - new_height - top_padding
    left_padding = (target_size - new_width) // 2
    right_padding = target_size - new_width - left_padding

    # 在图像周围添加黑色填充
    padded_image = cv2.copyMakeBorder(resized_image, top_padding, bottom_padding, left_padding, right_padding,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image


# 示例用法
image_path = 'IMG_5116.JPG'
image = cv2.imread(image_path)
padded_image = resize_and_pad_image(image)

# 保存或显示图片
cv2.imwrite('padded1024_image.jpg', padded_image)
# cv2.imshow('Padded Image', padded_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()