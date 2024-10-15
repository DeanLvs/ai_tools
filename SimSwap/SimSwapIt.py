import os
import torch
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from insightface_func.face_detect_crop_multi import Face_detect_crop as MultiFaceDetectCrop
from PIL import Image
from models.models import create_model
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
import glob
from parsing_model.model import BiSeNet

class FastCustomOptions:
    def __init__(self):
        # 初始化所有默认参数
        self.options = {
            'Arc_path': 'arcface_model/arcface_checkpoint.tar',
            'aspect_ratio': 1.0,
            'batchSize': 8,
            'checkpoints_dir': './checkpoints',
            'cluster_path': 'features_clustered_010.npy',
            'crop_size': 512,
            'data_type': 32,
            'dataroot': './datasets/cityscapes/',
            'display_winsize': 512,
            'engine': None,
            'export_onnx': None,
            'feat_num': 3,
            'fineSize': 512,
            'fp16': False,
            'gpu_ids': [0],
            'how_many': 50,
            'id_thres': 0.6, #身份阈值
            'image_size': 512,
            'input_nc': 3,
            'instance_feat': False,
            'isTrain': False,
            'label_feat': False,
            'label_nc': 0,
            'latent_size': 1024, #这个参数控制潜在变量的大小，当前为 512。你可以尝试不同的潜在变量大小，如 256 或 1024，看是否能改善生成质量。
            'loadSize': 2048,
            'load_features': False,
            'local_rank': 0,
            'max_dataset_size': float("inf"),
            'multisepcific_dir': './demo_file/multispecific',
            'nThreads': 2,
            'n_blocks_global': 6,
            'n_blocks_local': 3,
            'n_clusters': 10,
            'n_downsample_E': 4,
            'n_downsample_global': 3,
            'n_local_enhancers': 1,
            'name': 'people',
            'nef': 512 * 16,
            'netG': 'global',
            'ngf': 2048 * 16, # 当前设置为 64 和 16，可以尝试增加这些值来增强模型的表现力。比如，将 ngf 设置为 128，nef 设置为 32。
            'niter_fix_global': 0,
            'no_flip': False,
            'no_instance': False,
            'no_simswaplogo': False,
            'norm': 'batch',
            'norm_G': 'spectralspadesyncbatch3x3',
            'ntest': float("inf"),
            'onnx': None,
            'output_nc': 3,
            'output_path': './output/',
            'phase': 'test',
            'pic_a_path': 'G:/swap_data/ID/elon-musk-hero-image.jpeg',
            'pic_b_path': './demo_file/multi_people.jpg',
            'pic_specific_path': './crop_224/zrf.jpg',
            'resize_or_crop': 'scale_width',
            'results_dir': './results/',
            'semantic_nc': 3,
            'serial_batches': False,
            'temp_path': './temp_results',
            'tf_log': False,
            'use_dropout': False,
            'use_encoded_image': False,
            'use_mask': True,
            'verbose': False,
            'video_path': 'G:/swap_data/video/HSB_Demo_Trim.mp4',
            'which_epoch': 550000,
            'continue_train':True,
            'load_pretrain': True
        }

        # 将字典中的每个键值对设置为类属性
        for key, value in self.options.items():
            setattr(self, key, value)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.options:
                self.options[key] = value
                setattr(self, key, value)
            else:
                raise KeyError(f"Parameter {key} is not a valid option.")

    def display(self):
        for key, value in self.options.items():
            print(f"{key}: {value}")

    def get_param(self, key):
        return self.options.get(key, None)


transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)
class Options:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class SimSwap:
    def __init__(self):
        if not hasattr(self, 'model'):
            # 使用方法
            opt = FastCustomOptions()
            self.start_epoch, self.epoch_iter = 1, 0
            self.crop_size = opt.crop_size
            torch.nn.Module.dump_patches = True
            if opt.crop_size == 512:
                opt.which_epoch = 550000
                opt.name = '512'
                mode = 'ffhq'
            else:
                mode = 'None'
            self.opt = opt  # 将传入的 opt 赋值给类的属性
            self.logoclass = watermark_image('./simswaplogo/simswaplogo.png')
            model = create_model(opt)
            model.eval()

            self.spNorm = SpecificNorm()
            app = Face_detect_crop(name='antelope', root='./insightface_func/models')
            app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)

            self.model = model  # 使用传入的 opt 对象
            self.app = app

            mn_app = MultiFaceDetectCrop(name='antelope', root='./insightface_func/models')
            mn_app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)
            self.mn_app = mn_app


    def swap_faces(self, img_b_whole, img_a_whole=None):
        with torch.no_grad():
            img_a_align_crop, _ = self.app.get(img_a_whole, self.crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

            # convert numpy to tensor
            img_id = img_id.cuda()

            # create latent id
            img_id_downsample = F.interpolate(img_id, size=(112, 112))
            latend_id = self.model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)

            ############## Forward Pass ######################



            img_b_align_crop_list, b_mat_list = self.app.get(img_b_whole, self.crop_size)
            # detect_results = None
            swap_result_list = []

            b_align_crop_tenor_list = []

            for b_align_crop in img_b_align_crop_list:
                b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None, ...].cuda()

                swap_result = self.model(None, b_align_crop_tenor, latend_id, None, True)[0]
                swap_result_list.append(swap_result)
                b_align_crop_tenor_list.append(b_align_crop_tenor)

            if self.opt.use_mask:
                n_classes = 19
                net = BiSeNet(n_classes=n_classes)
                net.cuda()
                save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
                net.load_state_dict(torch.load(save_pth))
                net.eval()
            else:
                net = None
            img_ru = reverse2wholeimage(b_align_crop_tenor_list, swap_result_list, b_mat_list, self.crop_size, img_b_whole, self.logoclass, None, self.opt.no_simswaplogo,
                               pasring_model=net, use_mask=self.opt.use_mask, norm=self.spNorm)

            print(' ')

            print('************ Done ! ************')
        return img_ru


    def swap_faces_mn(self, pic_b_path, output_path, multisepcific_dir):
        source_specific_id_nonorm_list = []
        source_path = os.path.join(multisepcific_dir, 'SRC_*')
        source_specific_images_path = sorted(glob.glob(source_path))
        mse = torch.nn.MSELoss().cuda()
        for source_specific_image_path in source_specific_images_path:
            specific_person_whole = cv2.imread(source_specific_image_path)
            specific_person_align_crop, _ = self.mn_app.get(specific_person_whole, self.crop_size)
            specific_person_align_crop_pil = Image.fromarray(
                cv2.cvtColor(specific_person_align_crop[0], cv2.COLOR_BGR2RGB))
            specific_person = transformer_Arcface(specific_person_align_crop_pil)
            specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1],
                                                   specific_person.shape[2])
            # convert numpy to tensor
            specific_person = specific_person.cuda()
            # create latent id
            specific_person_downsample = F.interpolate(specific_person, size=(112, 112))
            specific_person_id_nonorm = self.model.netArc(specific_person_downsample)
            source_specific_id_nonorm_list.append(specific_person_id_nonorm.clone())

        # The person who provides id information (list)
        target_id_norm_list = []
        target_path = os.path.join(multisepcific_dir, 'DST_*')
        target_images_path = sorted(glob.glob(target_path))

        for target_image_path in target_images_path:
            img_a_whole = cv2.imread(target_image_path)
            img_a_align_crop, _ = self.mn_app.get(img_a_whole, self.crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            # convert numpy to tensor
            img_id = img_id.cuda()
            # create latent id
            img_id_downsample = F.interpolate(img_id, size=(112, 112))
            latend_id = self.model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
            target_id_norm_list.append(latend_id.clone())

        assert len(target_id_norm_list) == len(
            source_specific_id_nonorm_list), "The number of images in source and target directory must be same !!!"

        ############## Forward Pass ######################

        pic_b = pic_b_path
        img_b_whole = cv2.imread(pic_b)

        img_b_align_crop_list, b_mat_list = self.mn_app.get(img_b_whole, self.crop_size)
        # detect_results = None
        swap_result_list = []

        id_compare_values = []
        b_align_crop_tenor_list = []
        for b_align_crop in img_b_align_crop_list:

            b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None, ...].cuda()

            b_align_crop_tenor_arcnorm = self.spNorm(b_align_crop_tenor)
            b_align_crop_tenor_arcnorm_downsample = F.interpolate(b_align_crop_tenor_arcnorm, size=(112, 112))
            b_align_crop_id_nonorm = self.model.netArc(b_align_crop_tenor_arcnorm_downsample)

            id_compare_values.append([])
            for source_specific_id_nonorm_tmp in source_specific_id_nonorm_list:
                id_compare_values[-1].append(
                    mse(b_align_crop_id_nonorm, source_specific_id_nonorm_tmp).detach().cpu().numpy())
            b_align_crop_tenor_list.append(b_align_crop_tenor)

        id_compare_values_array = np.array(id_compare_values).transpose(1, 0)
        min_indexs = np.argmin(id_compare_values_array, axis=0)
        min_value = np.min(id_compare_values_array, axis=0)

        swap_result_list = []
        swap_result_matrix_list = []
        swap_result_ori_pic_list = []

        for tmp_index, min_index in enumerate(min_indexs):
            if min_value[tmp_index] < self.opt.id_thres:
                swap_result = \
                self.model(None, b_align_crop_tenor_list[tmp_index], target_id_norm_list[min_index], None, True)[0]
                swap_result_list.append(swap_result)
                swap_result_matrix_list.append(b_mat_list[tmp_index])
                swap_result_ori_pic_list.append(b_align_crop_tenor_list[tmp_index])
            else:
                pass

        if len(swap_result_list) != 0:

            if self.opt.use_mask:
                n_classes = 19
                net = BiSeNet(n_classes=n_classes)
                net.cuda()
                save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
                net.load_state_dict(torch.load(save_pth))
                net.eval()
            else:
                net = None

            reverse2wholeimage(swap_result_ori_pic_list, swap_result_list, swap_result_matrix_list, self.crop_size,
                               img_b_whole, self.logoclass, \
                               os.path.join(output_path, 'result_whole_swap_multispecific.jpg'), self.opt.no_simswaplogo,
                               pasring_model=net, use_mask=self.opt.use_mask, norm=self.spNorm)

            print(' ')

            print('************ Done ! ************')

        else:
            print('The people you specified are not found on the picture: {}'.format(pic_b))


# Example usage
if __name__ == "__main__":
    # 使用方法
    opt = FastCustomOptions()

    simswap = SimSwap()

    # Single face swap example
    img_b_whole = cv2.imread('56b9007efbf74157922f7537ca7de9dc.png')
    img_a_whole = cv2.imread('IMG_5003.png')

    final_img = simswap.swap_faces(img_b_whole=img_b_whole, img_a_whole=img_a_whole)
    cv2.imwrite('ttttttt.png', final_img)
    # Specific multi-face swap example
    # simswap.swap_faces(pic_b_path='./demo_file/multi_people.jpg', multispecific_dir='./demo_file/multispecific', output_path='./output/multi_result.png', mode="specific")