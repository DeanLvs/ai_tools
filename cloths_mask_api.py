import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as nnf
import threading
import matplotlib.colors as mcolors
import cv2
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

        self.device = "cuda" if torch.cuda.is_available() else "cpu" # sayeed99/segformer-b3-fashionsayeed99/segformer_b3_clothes
        self.processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
        self.segmentation_model = SegformerForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
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
        mask[np.isin(pred_seg, [4, 5, 6, 7, 8, 17])] = 255
        back_mask[np.isin(pred_seg, [1, 2, 3, 9, 10, 12, 13, 14, 15, 16])] = 255
        # mask[np.isin(pred_seg, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 20, 21, 22, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46])] = 255
        # back_mask[np.isin(pred_seg, [0])] = 255
        # 将NumPy数组转换为PIL图像
        mask = Image.fromarray(mask)
        back_mask = Image.fromarray(back_mask)

        print(f'finish split close mask {self.device}')
        return mask, back_mask, segmentation_image_pil

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import io
import numpy as np
import pickle
import asyncio,sys,os

def dilate_mask_by_percentage(mask, percentage, max_iterations=300, step_size=2):
    """
    根据整体像素的百分比扩张掩码。
    :param mask: 输入掩码（NumPy 数组）
    :param percentage: 需要扩张的百分比（0-100）
    :param max_iterations: 扩张的最大迭代次数（默认100）
    :return: 扩张后的掩码
    """
    # 计算目标扩张面积
    total_pixels = mask.size
    target_pixels = total_pixels * (percentage / 100.0)

    # 初始掩码的非零像素数量
    initial_non_zero = np.count_nonzero(mask)

    # 计算每次迭代后的扩张像素数量
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = mask.copy()
    last_non_zero = initial_non_zero

    for i in range(max_iterations):
        # 扩张掩码
        dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=step_size)
        current_non_zero = np.count_nonzero(dilated_mask)

        # 检查扩张是否达到目标像素数量
        if current_non_zero - initial_non_zero >= target_pixels:
            break

        last_non_zero = current_non_zero

    return np.clip(dilated_mask, 0, 255).astype(np.uint8)

def convert_to_ndarray(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError("Unsupported image type")

def re_auto_mask(file_path):
    print(f"begin check close mask: {file_path}")
    # 生成两个掩码
    mask_generator = MaskGenerator()
    image_pil = Image.open(file_path).convert("RGB")
    # 图片
    close_mask,not_close_mask,segmentation_image_pil = mask_generator.generate_mask_and_seg(image_pil)
    # 转换掩码为 ndarrays
    print(f'begin close mask convert_to_ndarray')
    # np数组
    densepose_mask_ndarray = convert_to_ndarray(close_mask)
    print(f'finish close mask convert_to_ndarray')
    # densepose_mask_ndarray = dilate_mask_np(densepose_mask_ndarray, iterations=8)
    print(f'begin dilate_mask_by_percentage close mask')
    # np数组
    densepose_mask_ndarray = dilate_mask_by_percentage(densepose_mask_ndarray, 6)
    print(f'finish dilate_mask_by_percentage close mask')
    # human_mask_ndarray = convert_to_ndarray(humanMask)
    # 合并掩码
    # merged_mask = merge_masks(densepose_mask_ndarray, human_mask_ndarray)
    # 合并相掉
    # merged_mask = subtract_masks(densepose_mask_ndarray, human_mask_ndarray)
    print(f"finish check close mask: {file_path}")
    return densepose_mask_ndarray, close_mask, convert_to_ndarray(not_close_mask), segmentation_image_pil

app = FastAPI()

async def restart_program():
    """Restarts the current program."""
    print("Restarting program...")
    # await asyncio.sleep(5)  # 延迟5秒，确保响应已发送
    python_executable = sys.executable  # 使用当前 Python 的可执行文件路径
    os.execv(python_executable, [python_executable] + sys.argv)
@app.get("/re")
async def inpaint(
    tasks: BackgroundTasks
):
    print('re start open_pose es ----------------')
    # 重启示范内存
    tasks.add_task(restart_program)

@app.post("/inpaint")
async def inpaint(
    file_path: str = Form(...),  # 接收文件路径
):
    print('start open_pose es ----------------')
    try:
        densepose_mask_ndarray, close_mask, not_close_mask_array, segmentation_image_pil = re_auto_mask(file_path)
        # 将多个数据打包到一个字典中
        response_data = {
            'densepose_mask_ndarray': densepose_mask_ndarray,
            'close_mask': close_mask,
            'not_close_mask_array': not_close_mask_array,
            'segmentation_image_pil': segmentation_image_pil
        }
        # 将字典通过 pickle 序列化
        masks_io = io.BytesIO()
        pickle.dump(response_data, masks_io)
        masks_io.seek(0)
        torch.cuda.empty_cache()
        # 创建 StreamingResponse 返回字节流
        return_it = StreamingResponse(masks_io, media_type="application/octet-stream")
        print('finish open_pose es ----------------')
        return return_it
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3060)