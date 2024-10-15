import os
import re
import time
from glob import iglob
from io import BytesIO
from flux.model import Flux, FluxParams
import streamlit as st
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
import torch
from einops import rearrange
from PIL import ExifTags, Image
from torchvision import transforms
from transformers import pipeline
from huggingface_hub import hf_hub_download
from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    configs,
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_sft
)

NSFW_THRESHOLD = 0.85

def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        sd = load_sft(ckpt_path, device="cpu")  # 使用 CPU 加载权重文件
        model.load_state_dict(sd, strict=False)
        model.to(device)  # 然后将模型移动到目标设备
    return model

def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device="cpu")  # 使用 CPU 加载权重文件
        ae.load_state_dict(sd, strict=False)
        ae.to(device)  # 然后将模型移动到目标设备
    return ae

def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    clip = load_clip(device)
    model = load_flow_model(name, device="cuda" if offload else device)
    ae = load_ae(name, device="cuda" if offload else device)
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
    return model, ae, clip, nsfw_classifier

def get_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )
    img: torch.Tensor = transform(image)
    return img[None, ...]

def get_mask(mask_path: str) -> torch.Tensor:
    mask = Image.open(mask_path).convert("L")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    msk: torch.Tensor = transform(mask)
    return msk[None, ...]

@torch.inference_mode()
def generate_image(
    image_path: str,
    mask_path: str,
    prompt: str,
    model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
    output_dir: str = "output",
    width: int = 1360,
    height: int = 768,
    num_steps: int = 50,
    guidance: float = 3.5,
    seed: int = None,
    noising_strength: float = 0.8,
    save_samples: bool = True,
    add_sampling_metadata: bool = True,
):
    torch_device = torch.device(device)
    model, ae, clip, nsfw_classifier = get_models(
        model_name,
        device=torch_device,
        offload=offload,
        is_schnell=(model_name == "flux-schnell"),
    )

    init_image = get_image(image_path)
    mask_image = get_mask(mask_path)

    if init_image is None or mask_image is None:
        raise ValueError("Both image and mask are required")

    if offload:
        ae.encoder.to(torch_device)
    init_image = ae.encode(init_image.to(torch_device))
    mask_image = mask_image.to(torch_device)
    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()

    x = get_noise(
        1,
        height,
        width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=seed,
    )
    timesteps = get_schedule(
        num_steps,
        (x.shape[-1] * x.shape[-2]) // 4,
        shift=(model_name != "flux-schnell"),
    )
    t_idx = int((1 - noising_strength) * num_steps)
    t = timesteps[t_idx]
    timesteps = timesteps[t_idx:]
    x = t * x + (1.0 - t) * (init_image * mask_image.to(x.dtype))

    if offload:
        clip = clip.to(torch_device)
    inp = prepare(clip=clip, img=x, prompt=prompt)

    if offload:
        clip = clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    x = unpack(x.float(), height, width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    if offload:
        ae.decoder.cpu()
        torch.cuda.empty_cache()

    fn = os.path.join(output_dir, "generated.jpg")
    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]

    if nsfw_score < NSFW_THRESHOLD:
        buffer = BytesIO()
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;img2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = model_name
        if add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = prompt
        img.save(buffer, format="jpeg", exif=exif_data, quality=95, subsampling=0)

        img_bytes = buffer.getvalue()
        if save_samples:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(fn, "wb") as file:
                file.write(img_bytes)

        return img, img_bytes, seed
    else:
        raise ValueError("Generated image may contain NSFW content.")

if __name__ == "__main__":
    # 输入参数
    image_path = "/mnt/sessd/ai_tools/static/uploads/3fc3cd52ced292773a155b9e9f90933b.png"
    mask_path = "/mnt/sessd/ai_tools/static/uploads/big_mask_3fc3cd52ced292773a155b9e9f90933b.png"
    prompt = "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX' is painted over it in big, red brush strokes with visible texture"
    model_name = "flux-schnell"
    output_dir = "output"
    width = 1360
    height = 768
    num_steps = 50
    guidance = 3.5
    seed = 42
    noising_strength = 0.8

    # 调用函数生成图像
    try:
        img, img_bytes, used_seed = generate_image(
            image_path=image_path,
            mask_path=mask_path,
            prompt=prompt,
            model_name=model_name,
            output_dir=output_dir,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            noising_strength=noising_strength,
        )
        print(f"Image generated with seed: {used_seed}")
        # 你可以进一步处理生成的图像或保存图像
    except ValueError as e:
        print(f"Error: {e}")