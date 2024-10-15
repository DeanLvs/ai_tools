import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage

# 将 lama 项目路径添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'lama')))

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_img_to_modulo

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return ToTensor()(image).unsqueeze(0)

def load_mask(mask_path):
    mask = Image.open(mask_path).convert('L')
    return ToTensor()(mask).unsqueeze(0)

def save_image(image_tensor, output_path):
    image = image_tensor.squeeze(0)
    image = ToPILImage()(image)
    image.save(output_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    checkpoint_path = '/mnt/essd/big-lama/models/best.ckpt'  # 修改为实际的checkpoint路径
    model = load_checkpoint(checkpoint_path, strict=False, map_location=device)
    model = model.to(device).eval()

    # 加载输入图像和掩码
    input_image_path = '/mnt/sessd/ai_tools/static/uploads/IMG_4878.png'
    mask_image_path = '/mnt/sessd/ai_tools/static/uploads/big_mask_IMG_4878.png'
    output_image_path = 'output.jpg'

    image = load_image(input_image_path).to(device)
    mask = load_mask(mask_image_path).to(device)

    # 将图像和掩码调整为模型需要的格式
    unpad_to_size = image.shape[2:]
    batch = {
        'image': image * 2 - 1,
        'mask': mask
    }
    batch = move_to_device(batch, device)
    batch['image'] = pad_img_to_modulo(batch['image'], 8)
    batch['mask'] = pad_img_to_modulo(batch['mask'], 8)

    # 推理
    with torch.no_grad():
        batch['inpainted'] = model(batch)['inpainted']

    output = batch['inpainted']
    output = (output + 1) / 2  # 恢复到 [0, 1] 范围
    output = output[:, :, :unpad_to_size[0], :unpad_to_size[1]]

    # 保存结果图像
    save_image(output, output_image_path)

if __name__ == '__main__':
    main()