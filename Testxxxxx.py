import cv2
import base64
import requests
import numpy as np

# 读取图像
img = cv2.imread('56b9007efbf74157922f7537ca7de9dc.png')

# 将图像编码为 JPEG 格式
_, img_encoded = cv2.imencode('.png', img)
img_n = cv2.imread('IMG_5003.png')
_, img_encoded_2 = cv2.imencode('.png', img_n)

img_base64 = base64.b64encode(img_encoded).decode('utf-8')
img_base64_n = base64.b64encode(img_encoded_2).decode('utf-8')
# 发送图像数据给服务器
data = {'image_data': img_base64_n, 'image_data_n': img_base64}
response = requests.post('http://localhost:5000/process_image', json=data)

# 获取返回的处理后的图像数据
processed_image = response.json()['processed_image']

# 将 Base64 字符串解码为二进制数据
img_binary = base64.b64decode(processed_image)

# 将二进制数据转换为 NumPy 数组并解码为图像
img_np = np.frombuffer(img_binary, np.uint8)
processed_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
cv2.imwrite('xxxxxxx.png', processed_img)