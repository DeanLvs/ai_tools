import os, sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import imageio

from pixielib.pixie import PIXIE
# from pixielib.pixie_parallel import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
from pixielib.utils.tensor_cropper import transform_points
from PIL import ImageDraw,Image, ImageFilter
from skimage.transform import warp, AffineTransform


from skimage.transform import warp, AffineTransform
import cv2
import numpy as np

import cv2
import numpy as np
import cv2
import numpy as np
from skimage.transform import warp, AffineTransform

def recover_and_overlay_to_original(rendered_image, original_image, tform_matrix):
    """
    将生成的 3D 渲染图像 (rendered_image) 恢复到原始图像 (original_image) 的坐标系，并将其叠加到原图上。

    :param rendered_image: 渲染的 3D 模型图像 (图 C)，大小为 1024x1024。
    :param original_image: 原始的完整输入图像 A。
    :param tform_matrix: 用于将裁剪后的图 B 恢复到原图 A 的仿射变换矩阵。
    :return: 将 3D 渲染结果叠加后的图像。
    """
    # 将 tform_matrix 转换为 AffineTransform 对象
    tform = AffineTransform(matrix=tform_matrix)

    # 使用 tform 的逆变换将渲染结果从裁剪后的图 B 恢复到原图 A 的坐标系
    projected_image = warp(rendered_image, tform.inverse, output_shape=original_image.shape)

    # 将 3D 渲染的结果叠加到原始图像 A 上
    result_image = original_image.copy()

    # 通过掩码来避免覆盖掉黑色区域
    mask = (projected_image != 0)  # 避免渲染结果中的黑色区域覆盖原图像
    result_image[mask] = projected_image[mask]

    return result_image

def resize_image(image_path, max_size=1024, file_name=""): #
    image = Image.open(image_path+file_name)
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
    file_name = 'w700d1q75cms.jpg'
    resize_image(image_path=inputpath_org, max_size=1024, file_name=file_name)
    inputpath = savefolder  + file_name
    original_image = cv2.imread(inputpath)  # 读取原始图像
    # load test images
    testdata = TestData(inputpath, iscrop=True, body_detector='rcnn')

    # -- run PIXIE
    pixie_cfg.model.use_tex = False
    pixie = PIXIE(config=pixie_cfg, device=device)
    visualizer = Visualizer(render_size=1024, config=pixie_cfg, device=device,
                            rasterizer_type='pytorch3d')
    render_size = 1024
    use_deca = False
    saveObj = False
    saveParam = False
    deca_path = None
    savePred = False
    saveImages = True
    lightTex = False
    extractTex = False
    showWeight = True
    saveVis = True
    saveGif = False
    showParts = False
    for i, batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
        util.move_dict_to_device(batch, device)
        tform_matrix = batch['tform'].numpy()  # 获取仿射变换矩阵
        batch['image'] = batch['image'].unsqueeze(0)
        batch['image_hd'] = batch['image_hd'].unsqueeze(0)
        name = batch['name']
        os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # print(name)
        # frame_id = int(name.split('frame')[-1])
        # name = f'{frame_id:05}'

        data = {
            'body': batch
        }
        param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
        # param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=True)
        # only use body params to get smplx output. TODO: we can also show the results of cropped head/hands
        moderator_weight = param_dict['moderator_weight']
        codedict = param_dict['body']
        opdict = pixie.decode(codedict, param_type='body')
        opdict['albedo'] = visualizer.tex_flame2smplx(opdict['albedo'])
        # -- save results
        if lightTex:
            visualizer.light_albedo(opdict)
        if extractTex:
            visualizer.extract_texture(opdict, data['body']['image_hd'])
        if saveVis:
            if showWeight is False:
                moderator_weight = None
            visdict = visualizer.render_results(opdict, data['body']['image_hd'], overlay=True, moderator_weight=moderator_weight, use_deca=use_deca)
        if saveObj:
            visualizer.save_obj(os.path.join(savefolder, name, f'{name}.obj'), opdict)
        if saveParam:
            codedict['bbox'] = batch['bbox']
            util.save_pkl(os.path.join(savefolder, name, f'{name}_param.pkl'), codedict)
            np.savetxt(os.path.join(savefolder, name, f'{name}_bbox.txt'), batch['bbox'].squeeze())
        if savePred:
            util.save_pkl(os.path.join(savefolder, name, f'{name}_prediction.pkl'), opdict)
        if saveImages:
            for vis_name in visdict.keys():
                cv2.imwrite(os.path.join(savefolder, name, f'{name}_{vis_name}.jpg'),
                            util.tensor2image(visdict[vis_name][0]))
        result_image = recover_and_overlay_to_original(util.tensor2image(visdict['shape_images'][0]), original_image, tform_matrix)
        # 保存叠加后的图像
        cv2.imwrite(os.path.join(savefolder, f'{name}_rxxxxxxxxed.jpg'), result_image)

    print(f'-- please check the results in {savefolder}')
if __name__ == '__main__':
    main()