import sys
import argparse
import cv2, traceback
import torch
import numpy as np
from typing import List, Optional
import time, gc, io
from pydantic import BaseModel
import os
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse

from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import read_video, get_target, get_final_video, add_audio_from_another_video, \
    face_enhancement
from utils.inference.core import model_inference

from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions

class InputArgs:
    def __init__(self, source_paths=[], target_faces_paths=[],
                 target_image='', out_image_name='outImage.png',
                 target_video='', out_video_name='outVideo.mp4',
                 G_path='weights/G_unet_3blocks.pth', backbone='unet', num_blocks=3, batch_size=40,
                 crop_size=224, use_sr=False, similarity_th=0.15,
                 image_to_image=False):
        self.G_path = G_path
        self.backbone = backbone  # ['unet', 'linknet', 'resnet']
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.use_sr = use_sr
        self.similarity_th = similarity_th
        self.source_paths = source_paths
        self.target_faces_paths = target_faces_paths
        self.target_video = target_video
        self.out_video_name = out_video_name
        self.image_to_image = image_to_image
        self.target_image = target_image
        self.out_image_name = out_image_name


def init_models(args):
    # model for face cropping
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    # main model for generation
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512)
    G.eval()
    G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')))
    G = G.cuda()
    G = G.half()

    # arcface model to get face embedding
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc = netArc.cuda()
    netArc.eval()

    # model to get face landmarks
    handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)

    # model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
    if args.use_sr:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        # opt.which_epoch ='10_7'
        model = Pix2PixModel(opt)
        model.netG.train()
    else:
        model = None

    return app, G, netArc, handler, model

def center_image_on_black_background(image, target_size=(640, 640)):
    """
    如果图像小于目标大小640x640，则将其放在纯黑色背景的中心，并补齐较小的方向。
    如果宽或高都大于等于目标大小，则返回原始图像。

    参数:
    - image: 要处理的图像（作为 NumPy 数组）。
    - target_size: 背景图像的目标尺寸，默认值为 (640, 640)。

    返回:
    - 处理后的图像，大小为 target_size。
    """
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # 如果图像的宽度和高度都大于或等于目标尺寸，直接返回原图像
    if original_width >= target_width and original_height >= target_height:
        return image

    # 确定需要的最终图像尺寸（补齐小于640的方向）
    new_width = max(original_width, target_width)
    new_height = max(original_height, target_height)

    # 创建一个纯黑色的背景图像，大小为new_width x new_height
    black_background = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # 计算图像应该放置的位置，使其居中
    x_offset = (new_width - original_width) // 2
    y_offset = (new_height - original_height) // 2

    # 将原图像粘贴到黑色背景上
    black_background[y_offset:y_offset + original_height, x_offset:x_offset + original_width] = image

    return black_background

def main(args):
    app, G, netArc, handler, model = init_models(args)

    # get crops from source images
    print('List of source paths: ', args.source_paths)
    source = []
    try:
        for source_path in args.source_paths:
            img = cv2.imread(source_path)
            img = center_image_on_black_background(img)
            img = crop_face(img, app, args.crop_size)[0]
            source.append(img[:, :, ::-1])
    except TypeError:
        print("Bad source images!")
        exit()

    # get full frames from video
    if not args.image_to_image:
        full_frames, fps = read_video(args.target_video)
    else:
        target_full = cv2.imread(args.target_image)
        full_frames = [target_full]

    # get target faces that are used for swap
    set_target = True
    print('List of target paths: ', args.target_faces_paths)
    if not args.target_faces_paths:
        target = get_target(full_frames, app, args.crop_size)
        set_target = False
    else:
        target = []
        try:
            for target_faces_path in args.target_faces_paths:
                img = cv2.imread(target_faces_path)
                img = center_image_on_black_background(img)
                img = crop_face(img, app, args.crop_size)[0]
                target.append(img)
        except TypeError:
            print("Bad target images!")
            exit()

    start = time.time()
    final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(full_frames,
                                                                                       source,
                                                                                       target,
                                                                                       netArc,
                                                                                       G,
                                                                                       app,
                                                                                       set_target,
                                                                                       similarity_th=args.similarity_th,
                                                                                       crop_size=args.crop_size,
                                                                                       BS=args.batch_size)
    if args.use_sr:
        final_frames_list = face_enhancement(final_frames_list, model)
    result = None
    if not args.image_to_image:
        get_final_video(final_frames_list,
                        crop_frames_list,
                        full_frames,
                        tfm_array_list,
                        args.out_video_name,
                        fps,
                        handler)
        add_audio_from_another_video(args.target_video, args.out_video_name, "audio")
        print(f"Video saved with path {args.out_video_name}")
    else:
        result = get_final_image(final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, handler)
        # cv2.imwrite(args.out_image_name, result)
        # print(f'Swapped Image saved with path {args.out_image_name}')

    print('Total time: ', time.time() - start)
    return result, args.out_video_name
def api_run(source_paths, target_faces_paths, target_image, target_video):
    if target_image is not None and target_image != '':
        image_to_image = True
        target_video = ''
    else:
        target_image = ''
        image_to_image = False
    args = InputArgs(source_paths=source_paths, target_faces_paths=target_faces_paths, target_image=target_image,
                     target_video=target_video, image_to_image=image_to_image)
    main(args)

def swap_multi(target_image, source_paths, target_faces_paths):
    args = InputArgs(source_paths=source_paths, target_faces_paths=target_faces_paths, target_image=target_image,
                     target_video='', image_to_image=True)
    face_img, video_path = main(args)
    return face_img

def swap_multi_video(target_video, video_sace_path, source_paths, target_faces_paths):
    args = InputArgs(source_paths=source_paths, target_faces_paths=target_faces_paths, target_video=target_video,
                     out_video_name=video_sace_path, image_to_image=False)
    face_img, video_path = main(args)
    return video_path


if __name__ == "__main__":
    # final_img = swap_multi(data.pic_b, data.target_path_list, data.source_path_list)
    # swap_multi_video(data.pic_b, data.pic_save, data.target_path_list, data.source_path_list)
    # source_paths = ['/nvme0n1-disk/ghost/sber-swap/3fc3cd52ced292773a155b9e9f90933b.jpg']
    # target_faces_paths = []
    # target_image = '/nvme0n1-disk/ghost/sber-swap/target_p/0c5ccccc00000001111aaaa.png'
    # target_video = ''
    # api_run(source_paths, target_faces_paths, target_image, target_video)

    pic_b = '/nvme0n1-disk/ai_tools/static/uploads/7104360038/f6b8bb6f-00bf-4483-98a4-f6db8965dbd0.MP4';

    source_path_list = [
        '/nvme0n1-disk/ai_tools/static/uploads/7104360038/mf_0_e12effdf-aec1-46de-91ed-14ddf72ba026.png',
        '/nvme0n1-disk/ai_tools/static/uploads/7104360038/mf_1_e12effdf-aec1-46de-91ed-14ddf72ba026.png']
    target_path_list = [
        '/nvme0n1-disk/ai_tools/static/uploads/7104360038/mf_2_e12effdf-aec1-46de-91ed-14ddf72ba026.png',
        '/nvme0n1-disk/ai_tools/static/uploads/7104360038/mf_3_e12effdf-aec1-46de-91ed-14ddf72ba026.png']

    # source_path_list = ['/nvme0n1-disk/ai_tools/static/uploads/7104360038/mf_0_3324e426-d3cd-4206-a759-e5211e780077.png', '/nvme0n1-disk/ai_tools/static/uploads/7104360038/mf_1_3324e426-d3cd-4206-a759-e5211e780077.png']
    # target_path_list = ['/nvme0n1-disk/ai_tools/static/uploads/7104360038/mf_2_3324e426-d3cd-4206-a759-e5211e780077.png', '/nvme0n1-disk/ai_tools/static/uploads/7104360038/mf_2_3324e426-d3cd-4206-a759-e5211e780077.png']
    swap_multi_video(pic_b, 'x91011xxxxxxFalsexx.MP4', target_path_list, source_path_list)