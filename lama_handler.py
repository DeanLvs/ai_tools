#!/usr/bin/env python3

import logging
import os
import traceback
import gc
import sys
import threading
sys.path.append("/mnt/sessd/ai_tools/lama")
from saicinpainting.evaluation.utils import move_to_device
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from saicinpainting.training.trainers import load_checkpoint
import cv2
import PIL.Image as Image
import numpy as np

LOGGER = logging.getLogger(__name__)
def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img

def load_image_l(mask_img, return_orig=False):
    img = np.array(mask_img)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img

def pad_img_to_modulo(img, mod):
    if isinstance(img, torch.Tensor) and img.is_cuda:  # 检查是否是CUDA张量
        img = img.cpu()  # 将张量从GPU移动到CPU
    if isinstance(img, torch.Tensor):  # 确保它是张量
        img = img.numpy()  # 将张量转换为Numpy数组
    if img.ndim == 2:  # 如果是灰度图像
        img = img[None, ...]  # 添加一个新的通道维度
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


class LaMaModel:
    _instance = None
    _model = None
    _lock = threading.Lock()  # 线程锁

    def __new__(cls, *args, **kwargs):
        with cls._lock:  # 确保线程安全的单例创建
            if not cls._instance:
                cls._instance = super(LaMaModel, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        if not self._model:
            train_config_path = os.path.join(model_path, 'config.yaml')
            with open(train_config_path, 'r') as f:
                train_config = OmegaConf.create(yaml.safe_load(f))

            train_config.training_model.predict_only = True
            train_config.visualizer.kind = 'noop'

            checkpoint_path = os.path.join(model_path, 'models', 'best.ckpt')
            self._model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location="cpu")
            self._model = self._model.float()  # 转换模型权重为 FloatTensor
            self._model.to(device=device)
            self._model.freeze()
            print(f'lama use at ----- {device}')
        self.device = device

    def predict(self, img_path, mask_path, maskImag, refine=False, iterations=5):
        try:
            image = load_image(img_path, mode='RGB')
            if mask_path:
                mask = load_image(mask_path, mode='L')
            else:
                mask = load_image_l(maskImag)
            # 确保输入图像和掩码都在相同的设备上，并且类型一致 (float32)
            image = torch.tensor(image, device=self.device, dtype=torch.float32)
            mask = torch.tensor(mask, device=self.device, dtype=torch.float32)
            result_image = image.clone()
            for i in range(iterations if refine else 1):
                # 准备输入数据
                result = dict(image=result_image, mask=mask[None, ...])
                result['unpad_to_size'] = result['image'].shape[1:]
                result['image'] = pad_img_to_modulo(result['image'], 8)
                result['mask'] = pad_img_to_modulo(result['mask'], 8)
                batch = default_collate([result])
                # 进行推理
                with torch.no_grad():
                    batch = move_to_device(batch, self.device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = self._model(batch)
                    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]
                # 处理结果
                result_image = np.clip(cur_res * 255, 0, 255).astype('uint8')
                result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                # 显式释放内存
                del batch
                torch.cuda.empty_cache()
                gc.collect()
            return result_image, True
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            return None,False
        finally:
            # 显式释放内存
            if 'batch' in locals():
                del batch
            # 显式释放内存
            if 'image' in locals():
                del image
            if 'mask' in locals():
                del mask
            torch.cuda.empty_cache()
            gc.collect()



