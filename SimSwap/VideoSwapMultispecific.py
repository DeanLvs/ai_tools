
import cv2
import torch
import fractions
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.videoswap_multispecific import video_swap
import os
import numpy as np
import glob

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# detransformer = transforms.Compose([
#         transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
#         transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
#     ])

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

def swap_multi_video(video_path='', save_path='', source_path_list=[], target_path_list=[]):
    opt = TestOptions().parse()
    opt.no_simswaplogo = True
    opt.use_mask = True
    crop_size = opt.crop_size
    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    model = create_model(opt)
    model.eval()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640), mode=mode)

    # The specific person to be swapped(source)

    source_specific_id_nonorm_list = []

    source_specific_images_path = source_path_list
    with torch.no_grad():
        for source_specific_image_path in source_specific_images_path:
            specific_person_whole = cv2.imread(source_specific_image_path)
            specific_person_whole = center_image_on_black_background(specific_person_whole)
            # cv2.imwrite('specific_person_whole.png', specific_person_whole)
            specific_person_align_crop, _ = app.get(specific_person_whole,crop_size)
            specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0],cv2.COLOR_BGR2RGB)) 
            specific_person = transformer_Arcface(specific_person_align_crop_pil)
            specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1], specific_person.shape[2])
            # convert numpy to tensor
            specific_person = specific_person.cuda()
            #create latent id
            specific_person_downsample = F.interpolate(specific_person, size=(112,112))
            specific_person_id_nonorm = model.netArc(specific_person_downsample)
            source_specific_id_nonorm_list.append(specific_person_id_nonorm.clone())


        # The person who provides id information (list)
        target_id_norm_list = []

        target_images_path = target_path_list

        for target_image_path in target_images_path:
            img_a_whole = cv2.imread(target_image_path)
            img_a_whole = center_image_on_black_background(img_a_whole)
            # 使用 OpenCV 保存图像，格式为 PNG
            # cv2.imwrite('img_a_or_whole.png', img_a_whole)
            img_a_align_crop, _ = app.get(img_a_whole,crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            # convert numpy to tensor
            img_id = img_id.cuda()
            #create latent id
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
            target_id_norm_list.append(latend_id.clone())

        assert len(target_id_norm_list) == len(source_specific_id_nonorm_list), "The number of images in source and target directory must be same !!!"

        video_swap(video_path, target_id_norm_list,source_specific_id_nonorm_list, opt.id_thres, \
            model, app, save_path,temp_results_dir=opt.temp_path,no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size)
    return save_path
# if __name__ == '__main__':
#     f_s_f = ['/nvme0n1-disk/book_yes/static/uploads/7104360038/mf_0_23ea59c0a2f80b07731c812f357aa520.png']
#     f_t_f = ['/nvme0n1-disk/book_yes/static/uploads/7104360038/2966c535-d4b0-4593-9cba-f27352b08863.jpg'] #mf_1_23ea59c0a2f80b07731c812f357aa520.png
#     p='/nvme0n1-disk/book_yes/static/uploads/7104360038/p_video_62d4dfc1-ee76-4ff6-9b57-de03e578975b.MP4'
#     finally_pic = swap_multi_video('/nvme0n1-disk/book_yes/static/uploads/7104360038/62d4dfc1-ee76-4ff6-9b57-de03e578975b.MP4',p, f_s_f, f_t_f)