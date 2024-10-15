import os
import cv2
import sys
import traceback
sys.path.append("/mnt/sessd/ai_tools/lama")
import io
import torch
import yaml
from omegaconf import OmegaConf
from saicinpainting.training.trainers import load_checkpoint
from torch.utils.data._utils.collate import default_collate
from saicinpainting.evaluation.utils import move_to_device
import numpy as np
from PIL import Image
import gc
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import asyncio
app = FastAPI()

def load_lama_model(model_dir='/mnt/sessd/ai_tools/big-lama', device='cuda'):
    """
    直接加载 LaMa 模型用于推理
    :param model_dir: 模型所在的目录，包括 config.yaml 和权重文件 (.ckpt)
    :param device: 使用的设备 ('cpu' 或 'cuda')
    :return: 已加载的 LaMa 模型
    """
    print(f'------加载 lama 到----- {device}')
    # 1. 加载 config.yaml
    config_path = os.path.join(model_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    # 2. 加载权重 big-lama.ckpt
    checkpoint_path = os.path.join(model_dir, 'models', 'best.ckpt')
    model = load_checkpoint(config, checkpoint_path, map_location=device, strict=False)
    # 将模型移动到指定设备
    model = model.to(device)
    model = model.float()
    model.eval()
    print(f'------加载完成 lama 到----- {device}')
    return model

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

def load_image_l(mask_img, return_orig=False):
    img = np.array(mask_img)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img
def inpaint_image(model, image_path, maskImag, device='cuda', refine=False, iterations=5):
    """
    使用 LaMa 模型进行图像修复
    :param model: LaMa 模型
    :param image: 输入图像，numpy 数组
    :param mask: 修复掩码，numpy 数组
    :return: 修复后的图像，numpy 数组
    """
    print(f'------使用 lama 推理中----- {device}')
    try:
        image = load_image(image_path, mode='RGB')
        mask = load_image_l(maskImag)
        # 将输入图像和掩码都放在指定设备上
        image = torch.tensor(image, device=device, dtype=torch.float32)
        mask = torch.tensor(mask, device=device, dtype=torch.float32)
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
                batch = move_to_device(batch, device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)
                cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]
            # 处理结果
            result_image = np.clip(cur_res * 255, 0, 255).astype('uint8')
            result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            # 显式释放内存
            print(f'------ 循环 使用 lama 推理中----- {device}')
            del batch
            print(f'------ 循环 使用 lama 推理中--清理 batch 释放空间--- {device}')
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        return None, False
    return result_image, True

def lama_inpaint_with_memory_mask(image_path, mask_array):
    """
    LaMa 图像修复的主函数，支持内存中直接传递的图像和掩码
    :param image_path: 输入图像路径
    :param mask_array: 输入的掩码，numpy 数组
    :param output_path: 输出图像保存路径
    """

    # 加载 LaMa 模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_lama_model(device=device)
    print(f"Inpainting complete. Output saved")
    return inpaint_image(model, image_path, mask_array)

async def restart_program():
    """Restarts the current program."""
    print("Restarting program...")
    await asyncio.sleep(1)  # 延迟5秒，确保响应已发送
    python_executable = sys.executable  # 使用当前 Python 的可执行文件路径
    os.execv(python_executable, [python_executable] + sys.argv)

@app.post("/inpaint")
async def inpaint(
    file_path: str = Form(...),  # 接收文件路径
    mask_clear: UploadFile = File(...)  # 接收 mask_clear 图像
):
    # 加载 mask_clear 图像
    mask_img = Image.open(mask_clear.file)
    # LaMa 修复
    try:
        result_image, reslt_t = lama_inpaint_with_memory_mask(file_path, mask_img)
        # 将 numpy 数组转换为 PIL.Image 对象
        result_image_pil = Image.fromarray(result_image)
        # 将 PIL.Image 保存到字节流
        img_io = io.BytesIO()
        result_image_pil.save(img_io, format='PNG')  # 这里你可以选择其他格式
        img_io.seek(0)
        # 重启示范内存
        asyncio.create_task(restart_program())
        # 返回 StreamingResponse，以 PNG 格式返回图像
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
