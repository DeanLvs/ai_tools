import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as nnf
import numpy as np
import threading
import matplotlib.colors as mcolors

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
        self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.segmentation_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.segmentation_model.to(self.device)

    def generate_mask(self, image):
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

        # back_mask = Image.new("L", image.size, 0)
        # # 创建掩码，找到衣服区域（上衣、裙子、裤子、连衣裙）
        # mask = Image.new("L", image.size, 0)
        # print(f'finish close mask {self.device}')
        # for y in range(pred_seg.shape[0]):
        #     for x in range(pred_seg.shape[1]):
        #         if pred_seg[y, x].item() in [4, 5, 6, 7]:  # 上衣、裙子、裤子、连衣裙
        #             mask.putpixel((x, y), 255)
        # print(f'finish not close mask {self.device}')
        # for y in range(pred_seg.shape[0]):
        #     for x in range(pred_seg.shape[1]): #0, 背景、脸,
        #         if pred_seg[y, x].item() in [1, 2, 3, 8, 9,10,12,13,14,15,16,17]:  # 帽子、头发、太阳镜,腰带,左鞋,右鞋,左腿,右腿,左臂,右臂,包,围巾
        #             back_mask.putpixel((x, y), 255)
        print(f'finish close mask {self.device}')
        # 创建掩码，找到衣服区域（上衣、裙子、裤子、连衣裙）
        mask = np.zeros(image.size[::-1], dtype=np.uint8)
        back_mask = np.zeros(image.size[::-1], dtype=np.uint8)

        # 使用NumPy向量化操作
        mask[np.isin(pred_seg, [4, 5, 6, 7])] = 255
        back_mask[np.isin(pred_seg, [1, 2, 3, 8, 9, 10, 12, 13, 14, 15, 16, 17])] = 255

        # 将NumPy数组转换为PIL图像
        mask = Image.fromarray(mask)
        back_mask = Image.fromarray(back_mask)

        print(f'finish split close mask {self.device}')
        return mask, back_mask

    def generate_mask_pixels(self, image):
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
        pred_seg = upsampled_logits.argmax(dim=1)[0]

        mask_pixels = []
        for y in range(pred_seg.shape[0]):
            for x in range(pred_seg.shape[1]):
                if pred_seg[y, x].item() in [4, 5, 6, 7]:  # 上衣、裙子、裤子、连衣裙
                    mask_pixels.append((x, y))

        return mask_pixels

    def generate_mask_f(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 获取分割结果
        with torch.no_grad():
            outputs = self.segmentation_model(**inputs)
        logits = outputs.logits.cpu() if self.device == "cpu" else outputs.logits

        # 上采样到输入图像大小
        upsampled_logits = nnf.interpolate(
            logits,
            size=image.size[::-1],  # PIL.Image size is (width, height)
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # 创建掩码，找到衣服区域（上衣、裙子、裤子、连衣裙）
        mask_array = np.isin(pred_seg, [4, 5, 6, 7]).astype(np.uint8) * 255
        return mask_array

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

        # 将NumPy数组转换为PIL图像
        mask = Image.fromarray(mask)
        back_mask = Image.fromarray(back_mask)

        print(f'finish split close mask {self.device}')
        return mask, back_mask, segmentation_image_pil