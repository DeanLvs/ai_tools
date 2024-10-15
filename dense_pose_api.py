import threading
import gc
import glob
import zipfile
import os
import sys
import torch
import cv2
import numpy as np
from controlnet_aux import HEDdetector
from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image, _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from densepose import add_densepose_config
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.extractor import create_extractor
import pickle
import gc
from PIL import Image
import requests, io
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio, io
app = FastAPI()

parts_to_include_local = {
        "Torso_1": [1],
        "Torso_2": [2],
        "Upper_Leg_Right_7": [7],
        "Upper_Leg_Left_8": [8],
        "Upper_Leg_Right_9": [9],
        "Upper_Leg_Left_10": [10],
        "Lower_Leg_Right_11": [11],
        "Lower_Leg_Left_12": [12],
        "Lower_Leg_Right_13": [13, 6],
        "Lower_Leg_Left_14": [14, 5],
        "Upper_Arm_Left_15": [15],
        "Upper_Arm_Right_16": [16],
        "Upper_Arm_Left_17": [17],
        "Upper_Arm_Right_18": [18],
        "Lower_Arm_Left_19": [19],
        "Lower_Arm_Right_20": [20],
        "Lower_Arm_Left_21": [21, 4],
        "Lower_Arm_Right_22": [22, 3],
        "exclude": [23,24], #3,4,5,6,
        "all_body": [1, 2, 23, 24, 15,16,17,18,19,20,21,22],
        "Torso_line": [15,16,17,18,19,20,21,22,3,4,23,24],
    }

parts_to_include_down_body = {
        "Torso_line": [15,16,17,18,19,20,21,22,3,4,5,6,23,24], # 1,2
        "left_line": [7,9,11,13],
        "right_line": [8,10,12,14]
    }
class DensePoseModel:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_file, model_file, device="cpu"):
        with cls._lock:  # 确保线程安全的单例创建
            if cls._instance is None:
                cls._instance = super(DensePoseModel, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file, model_file, device="cpu"):
        if self._initialized:
            return
        self.cfg = self.setup_config(config_file, model_file, device)
        self.predictor = DefaultPredictor(self.cfg)
        self.logger = setup_logger(name="DensePoseModel")
        self._initialized = True

    def setup_config(self, config_file, model_file, device):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_file
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        return cfg

    def generate_mask(self, img, parts_to_include=None):
        img_height, img_width = img.shape[:2]
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]

        # 使用 fine segmentation visualizer 生成掩码
        visualizer = DensePoseResultsFineSegmentationVisualizer(self.cfg)
        extractor = create_extractor(visualizer)
        data = extractor(outputs)
        # 创建一个全黑的掩码
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if parts_to_include is not None:
            for instance_idx, instance_data in enumerate(data[0]):  # 遍历所有实例
                labels = instance_data.labels.cpu().numpy()
                bbox = outputs.pred_boxes.tensor[instance_idx].cpu().numpy().astype(int)
                x0, y0, x1, y1 = bbox
                part_mask = np.zeros_like(labels, dtype=np.uint8)
                for part in parts_to_include:
                    part_mask = np.maximum(part_mask, (labels == part).astype(np.uint8) * 255)

                # 调整掩码大小以匹配原始图像
                if (x1 - x0) > 0 and (y1 - y0) > 0:
                    part_mask_resized = cv2.resize(part_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
                    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], part_mask_resized)
                else:
                    self.logger.warning(f"Invalid bounding box dimensions: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
        return mask

    def generate_m_mask(self, img, parts_to_include=None):
        img_height, img_width = img.shape[:2]
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]

        # 使用 fine segmentation visualizer 生成掩码
        visualizer = DensePoseResultsFineSegmentationVisualizer(self.cfg)
        extractor = create_extractor(visualizer)
        data = extractor(outputs)
        torch.cuda.empty_cache()
        gc.collect()
        # 创建字典来存储不同部分的掩码
        masks = {key: np.zeros((img_height, img_width), dtype=np.uint8) for key in parts_to_include}

        if parts_to_include is not None:
            for instance_idx, instance_data in enumerate(data[0]):  # 遍历所有实例
                labels = instance_data.labels.cpu().numpy()
                bbox = outputs.pred_boxes.tensor[instance_idx].cpu().numpy().astype(int)
                x0, y0, x1, y1 = bbox

                for key, parts in parts_to_include.items():
                    part_mask = np.zeros_like(labels, dtype=np.uint8)
                    for part in parts:
                        part_mask = np.maximum(part_mask, (labels == part).astype(np.uint8) * 255)

                    # 调整掩码大小以匹配原始图像
                    if (x1 - x0) > 0 and (y1 - y0) > 0:
                        part_mask_resized = cv2.resize(part_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
                        masks[key][y0:y1, x0:x1] = np.maximum(masks[key][y0:y1, x0:x1], part_mask_resized)
                    else:
                        self.logger.warning(f"Invalid bounding box dimensions: x0={x0}, y0={y0}, x1={x1}, y1={y1}")

        return masks

def handlerMaskAll(img_path, input_img=None, parts_to_include=None):
    if input_img is not None:
        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image = _apply_exif_orientation(input_img)
        img = convert_PIL_to_numpy(image, format="BGR")
    else:
        img = read_image(img_path, format="BGR")
    # 配置文件和模型文件路径
    config_file = "detectron2/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_WC2M_s1x.yaml"
    model_file = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2M_s1x/216245790/model_final_de6e7a.pkl"
    # 判断是否可以使用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # 初始化 DensePose 模型
    densepose_model = DensePoseModel(config_file, model_file, device)
    # 移除 ,23, 24
    ###fine
    # segmentation: 1, 2 = Torso, 3 = Right
    # Hand, 4 = Left
    # Hand,
    # 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
    # 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right,
    # 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
    # 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left,
    # 20, 22 = Lower Arm Right, 23, 24 = Head
    # parts_to_include = [3,4,5,6]
    body_to_include = [1, 2]
    leg_to_include  = [7,9, 8,10, 11,13, 12,14]
    arm_to_include = [15,17, 16,18, 19,21, 20,22]
    head_to_include = [23,24]
    hand_to_include = [3, 4]
    foot_to_include = [5, 6]
    mask = densepose_model.generate_m_mask(img, parts_to_include=parts_to_include)
    torch.cuda.empty_cache()
    gc.collect()
    return mask

async def restart_program():
    """Restarts the current program."""
    print("Restarting program...")
    python_executable = sys.executable  # 使用当前 Python 的可执行文件路径
    os.execv(python_executable, [python_executable] + sys.argv)

def req_3d_pose(file_path):
    # 发起 HTTP POST 请求
    url = "http://localhost:3001/inpaint"
    data = {
        'file_path': file_path
    }
    response = requests.post(url, data=data)

    # 处理响应的图像
    if response.status_code == 200:
        img_io_re = io.BytesIO(response.content)
        img_pil = Image.open(img_io_re)
        print(f'-----suc return lam---------')
        return img_pil
    print(f'-----error return lam---------')
    return None

class HEDdetectorSingleton:
    _instance = None
    _lock = threading.Lock()  # A lock object to ensure thread-safe singleton access
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check locking to avoid race conditions
                    print(f'------加载 lama 到-----')
                    model = HEDdetector.from_pretrained('lllyasviel/Annotators')
                    print(f'------加载完成 lama 到-----')
                    cls._instance = model
        return cls._instance

@app.get("/re")
async def inpaint(
    tasks: BackgroundTasks
):
    print('re start depose es ----------------')
    # 重启示范内存
    url = "http://localhost:3001/re"
    requests.get(url)
    tasks.add_task(restart_program)

@app.post("/inpaint")
async def inpaint(
    file_path: str = Form(...)
):
    # dense 分析
    try:
        # 获取 file_path 对应图片的大小
        original_image = Image.open(file_path)
        original_size = original_image.size  # 获取原始图片的大小 (width, height)
        masks = handlerMaskAll(file_path, input_img = original_image, parts_to_include = parts_to_include_local)
        d_pose = req_3d_pose(file_path)
        print(f'success d_pose')
        # 调整 d_pose 大小，使其与 file_path 的原始图片大小一致
        d_pose_resized = d_pose.resize(original_size, Image.LANCZOS)

        print(f'success d_pose resized')

        # 计算膨胀大小（图像最大尺寸的 5%）
        dilation_size = int(max(original_size) * 0.015)

        # 将 d_pose_resized 转换为字节流
        d_pose_io = io.BytesIO()
        d_pose_resized.save(d_pose_io, format="PNG")
        d_pose_io.seek(0)

        print(f'success d_pose resized')
        masks_down = handlerMaskAll(file_path, input_img=d_pose_resized, parts_to_include=parts_to_include_down_body)
        print(f'success masks_down')
        masks['Torso_line'] = masks_down['Torso_line']
        # masks['left_line'] = masks_down['left_line']
        # masks['right_line'] = masks_down['right_line']
        # 将掩码转换为字节流 (pickle 序列化)
        masks_io = io.BytesIO()
        pickle.dump(masks, masks_io)
        masks_io.seek(0)

        processor = HEDdetectorSingleton()
        d_3d_canny_resized = processor(d_pose_resized, scribble=False)
        d_3d_canny_resized = d_3d_canny_resized.resize(original_size, Image.LANCZOS)

        # 创建膨胀的结构元素
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))

        keyA_mask = masks['Torso_line']
        # 膨胀 keyA 区域
        dilated_keyA = cv2.dilate(keyA_mask, structuring_element)

        # 将 d_3d_canny_resized 转换为 numpy 数组进行修改
        d_3d_canny_resized_array = np.array(d_3d_canny_resized)

        # 在 d_3d_canny_resized 中将膨胀后的 keyA 区域设为黑色
        d_3d_canny_resized_array[dilated_keyA > 0] = 0

        # 将修改后的数组转换回图像
        d_3d_canny_resized_modified = Image.fromarray(d_3d_canny_resized_array)

        print(f'start run 3d canny care it')
        # 将 d_pose_resized 转换为字节流
        d_3d_canny_resized_io = io.BytesIO()
        d_3d_canny_resized_modified.save(d_3d_canny_resized_io, format="PNG")
        d_3d_canny_resized_io.seek(0)
        print(f'finish run 3d canny care it')
        # 创建 ZIP 文件并将 d_pose 和 masks 打包
        zip_io = io.BytesIO()
        with zipfile.ZipFile(zip_io, mode="w") as zf:
            # 添加 d_pose_resized 到 ZIP
            zf.writestr("d_pose_resized.png", d_pose_io.getvalue())
            zf.writestr("d_3d_canny_resized.png", d_3d_canny_resized_io.getvalue())
            # 添加 masks 到 ZIP
            zf.writestr("masks.pkl", masks_io.getvalue())

        zip_io.seek(0)
        torch.cuda.empty_cache()
        # 返回 ZIP 文件作为响应
        re_it = StreamingResponse(zip_io, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=result.zip"})
        # 返回字节流响应
        return re_it
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9080)
