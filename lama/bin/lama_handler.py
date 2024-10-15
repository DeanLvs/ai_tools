#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import traceback

os.environ['TORCH_HOME'] = '/mnt/essd/ai_tools/torch_home'
import sys

sys.path.append("/mnt/essd/ai_tools/lama")

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
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
def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


class LaMaModel:
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
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
            self._model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device)
            self._model.freeze()
            self._model.to(device)

        self.device = device

    def predict(self, img_path, mask_path, refine=False):
        image = load_image(img_path, mode='RGB')
        mask = load_image(mask_path, mode='L')
        result = dict(image=image, mask=mask[None, ...])
        # if self.scale_factor is not None:
        #     result['image'] = scale_image(result['image'], self.scale_factor)
        #     result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)
        # if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
        result['unpad_to_size'] = result['image'].shape[1:]
        result['image'] = pad_img_to_modulo(result['image'], 8)
        result['mask'] = pad_img_to_modulo(result['mask'], 8)
        batch = default_collate([result])
        print("fin1")
        with torch.no_grad():
            batch = move_to_device(batch, self.device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = self._model(batch)
            cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        print("fin6")
        cv2.imwrite('ressult12221212.png', cur_res)
        print("fin7")



# 使用示例
model_instance = LaMaModel(model_path='/mnt/essd/ai_tools/Inpaint-Anything/big-lama')
model_instance.predict('/mnt/essd/ai_tools/lama/LaMa_test_images/000068.png', '/mnt/essd/ai_tools/lama/LaMa_test_images/000068_mask.png')
print("fin8")

