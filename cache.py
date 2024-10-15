import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as nnf
import threading
import matplotlib.colors as mcolors
import cv2
import numpy as np

class MaskGenerator:
    _instance = None
    _lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        with cls._lock:  # 确保线程安全的单例创建
            if not cls._instance:
                cls._instance = super(MaskGenerator, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b3-fashion")
        self.segmentation_model = SegformerForSemanticSegmentation.from_pretrained("sayeed99/segformer-b3-fashion")
        self.segmentation_model.to(self.device)

    def generate_mask_and_seg(self, image):
        print(f'begin close mask {self.device}')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 获取分割结果
        with torch.no_grad():
            outputs = self.segmentation_model(**inputs)
        if self.device == "cpu":
            logits = outputs.logits.cpu()
        else:
            logits = outputs.logits

        # 上采样到输入图像大小
        upsampled_logits = nnf.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # 定义类别对应的颜色（你可以根据需要修改颜色）
        colors = list(mcolors.TABLEAU_COLORS.values())  # 使用 Tableau 颜色
        num_classes = len(colors)
        # 创建彩色分割图像
        segmentation_image = np.zeros((*pred_seg.shape, 3), dtype=np.uint8)
        # 为每个类别分配不同的颜色
        for i in range(num_classes):
            segmentation_image[pred_seg == i] = np.array(mcolors.to_rgb(colors[i])) * 255

        # 将彩色分割图像转换为 PIL 图像
        segmentation_image_pil = Image.fromarray(segmentation_image.astype(np.uint8))

        print(f'finish close mask {self.device}')
        # 创建掩码，找到衣服区域（上衣、裙子、裤子、连衣裙）
        mask = np.zeros(image.size[::-1], dtype=np.uint8)
        back_mask = np.zeros(image.size[::-1], dtype=np.uint8)

        # 使用NumPy向量化操作
        mask[np.isin(pred_seg, [4, 5, 6, 7])] = 255
        back_mask[np.isin(pred_seg, [1, 2, 3, 8, 9, 10, 12, 13, 14, 15, 16, 17])] = 255
        # mask[np.isin(pred_seg, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 20, 21, 22, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 44, 45, 46])] = 255
        # back_mask[np.isin(pred_seg, [0])] = 255
        # 将NumPy数组转换为PIL图像
        mask = Image.fromarray(mask)
        back_mask = Image.fromarray(back_mask)

        print(f'finish split close mask {self.device}')
        return mask, back_mask, segmentation_image_pil

if __name__ == '__main__':
    file_path = "/mnt/sessd/ai_tools/static/uploads/0c5612f9-54fb-4b5c-95f1-475e55c31a0d/C5A33A03-D152-4745-81F2-CFCE8292E3CB.jpeg"
    mask_generator = MaskGenerator()
    image_pil = Image.open(file_path).convert("RGB")
    # 图片
    close_mask, not_close_mask, segmentation_image_pil = mask_generator.generate_mask_and_seg(image_pil)
