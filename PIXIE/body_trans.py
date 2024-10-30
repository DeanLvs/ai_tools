import os, sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import imageio
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
from PIL import Image


def resize_image(image_path, max_size=2048, file_name=""):
    image = Image.open(image_path + file_name)
    width, height = image.size

    if width > max_size or height > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        if image.mode == 'P':
            image = image.convert('RGB')
        image.save(f'/Users/dean/PycharmProjects/Body/PIXIE/{file_name}')


def main():
    device = 'cpu'
    inputpath_org = '/Users/dean/Desktop/uploads/'
    savefolder = '/Users/dean/PycharmProjects/Body/PIXIE/'
    source_file_name = 'w700d1q75cms.jpg'  # 源图像
    target_file_name = 'w700d1q75cms.jpg'  # 目标图像

    # Resize and load images
    resize_image(image_path=inputpath_org, max_size=1000, file_name=source_file_name)
    resize_image(image_path=inputpath_org, max_size=1000, file_name=target_file_name)
    source_inputpath = savefolder + source_file_name
    target_inputpath = savefolder + target_file_name

    # Load test images
    source_testdata = TestData(source_inputpath, iscrop=True, body_detector='rcnn')
    target_testdata = TestData(target_inputpath, iscrop=True, body_detector='rcnn')

    # Initialize PIXIE
    pixie_cfg.model.use_tex = False
    pixie = PIXIE(config=pixie_cfg, device=device)
    visualizer = Visualizer(render_size=1024, config=pixie_cfg, device=device, rasterizer_type='pytorch3d')

    # -- Step 1: Extract 3D parameters from source image
    source_batch = source_testdata[0]
    util.move_dict_to_device(source_batch, device)
    source_batch['image'] = source_batch['image'].unsqueeze(0)
    source_batch['image_hd'] = source_batch['image_hd'].unsqueeze(0)

    source_data = {'body': source_batch}
    source_param_dict = pixie.encode(source_data, threthold=True, keep_local=True, copy_and_paste=False)
    source_codedict = source_param_dict['body']

    # -- Step 2: Fit 3D parameters into target image
    for i, target_batch in enumerate(tqdm(target_testdata, dynamic_ncols=True)):
        util.move_dict_to_device(target_batch, device)
        target_batch['image'] = target_batch['image'].unsqueeze(0)
        target_batch['image_hd'] = target_batch['image_hd'].unsqueeze(0)
        name = target_batch['name']
        os.makedirs(os.path.join(savefolder, name), exist_ok=True)

        # Prepare target data
        target_data = {'body': target_batch}
        target_param_dict = pixie.encode(target_data, threthold=True, keep_local=True, copy_and_paste=False)
        target_codedict = target_param_dict['body']

        # Transfer 3D information from source image to target image
        target_codedict['shape'] = source_codedict['shape']  # Transfer shape
        target_codedict['pose'] = source_codedict['pose']  # Transfer pose

        # Decode and render the target image with source 3D information
        opdict = pixie.decode(target_codedict, param_type='body')
        opdict['albedo'] = visualizer.tex_flame2smplx(opdict['albedo'])
        visdict = visualizer.render_results(opdict, target_batch['image_hd'], overlay=True)

        # Save the rendered results
        cv2.imwrite(os.path.join(savefolder, f'{name}_transfer_vis.jpg'),
                    visualizer.visualize_grid(visdict, size=1024))

        print(f'-- Result saved for target image: {name}')


if __name__ == '__main__':
    main()