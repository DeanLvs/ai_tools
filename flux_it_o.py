from diffusers import FluxInpaintPipeline
import torch
from diffusers.utils import load_image

ckpt_id = "black-forest-labs/FLUX.1-schnell"
prompt = "sexy, person,woman,authenticity,natural asses,natural boobs"
# prompt = [prompt] * 2
# denoising
pipe = FluxInpaintPipeline.from_pretrained(
    ckpt_id,
    torch_dtype=torch.bfloat16,
)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.enable_sequential_cpu_offload() # offloads modules to CPU on a submodule level (rather than model level)
print('Max mem allocated (GB) while denoising:', torch.cuda.max_memory_allocated() / (1024 ** 3))
source = load_image("/mnt/sessd/ai_tools/static/uploads/59613e59-24c4-4e00-850f-7235b57397a2/0988ae0c890642d8a094ad8370a4df9d.jpeg")
mask = load_image("/mnt/sessd/ai_tools/static/uploads/59613e59-24c4-4e00-850f-7235b57397a2/with_torso_mask_0988ae0c890642d8a094ad8370a4df9d.jpeg")
image = pipe(prompt=prompt, image=source, mask_image=mask).images[0]
image.save("flux_inpainting.png")
