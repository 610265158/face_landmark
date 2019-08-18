


import os
import random
import cv2
import numpy as np
from functools import partial
import traceback
import copy

from lib.helper.logger import logger
from tensorpack.dataflow import DataFromList
from tensorpack.dataflow import BatchData, MultiThreadMapData, MultiProcessPrefetchData


from lib.dataset.augmentor.augmentation import Rotate_aug,\
                                        Affine_aug,\
                                        Mirror,\
                                        Padding_aug,\
                                        produce_heat_maps

from lib.dataset.augmentor.visual_augmentation import ColorDistort
from lib.dataset.headpose import get_head_pose
from train_config import config as cfg


class data_info(object):
    def __init__(self,img_root,txt):
        self.txt_file=txt
        self.root_path = img_root
        self.metas=[]


        self.read_txt()

    def read_txt(self):
        with open(self.txt_file) as _f:
            txt_lines=_f.readlines()

        for line in txt_lines:
            line=line.rstrip()

            _img_path=line.rsplit('|',1)[0]
            _label=line.rsplit('|',1)[-1]

            current_img_path=os.path.join(self.root_path,_img_path)
            current_img_label=_label
            self.metas.append([current_img_path,current_img_label])

            ###some change can be made here
        logger.info('the dataset contains %d images'%(len(txt_lines)))
        logger.info('the datasets contains %d samples'%(len(self.metas)))


    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas


class BaseDataIter():
    def __init__(self,img_root_path='',ann_file=None,training_flag=True):

        self.shuffle=True
        self.training_flag=training_flag

        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.thread_num = cfg.TRAIN.thread_num
        self.process_num = cfg.TRAIN.process_num
        self.buffer_size = cfg.TRAIN.buffer_size
        self.prefetch_size = cfg.TRAIN.prefetch_size


        self.dataset_list = self.parse_file(img_root_path, ann_file)

        self.ds=self.build_iter(self.dataset_list)


    def parse_file(self,im_root_path,ann_file):

        raise NotImplementedError("you need implemented the parse func for your data")


    def build_iter(self,samples):

        map_func=partial(self._map_func,is_training=self.training_flag)
        ds = DataFromList(samples, shuffle=True)

        ds = MultiThreadMapData(ds, self.thread_num, map_func, buffer_size=self.buffer_size)

        ds = BatchData(ds, self.num_gpu *  self.batch_size)
        ds = MultiProcessPrefetchData(ds, self.prefetch_size, self.process_num)
        ds.reset_state()
        ds = ds.get_data()
        return ds

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.ds)


    def _map_func(self,dp,is_training):

        raise NotImplementedError("you need implemented the map func for your data")




class FaceKeypointDataIter(BaseDataIter):
    def __init__(self, img_root_path='', ann_file=None, training_flag=True):

        self.color_augmentor = ColorDistort()

        ###init the base class at last !!
        super(FaceKeypointDataIter, self).__init__(img_root_path, ann_file, training_flag)

    def balance(self,anns):
        res_anns = copy.deepcopy(anns)

        lar_count = 0
        for ann in anns:
            label = ann[-1]
            label = np.array([label.split(' ')], dtype=np.float).reshape((-1, 2))
            bbox = np.array([np.min(label[:, 0]), np.min(label[:, 1]), np.max(label[:, 0]), np.max(label[:, 1])])
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]

            if bbox_width < 30 or bbox_height < 30:
                res_anns.remove(ann)

            left_eye_close = np.sqrt(
                np.square(label[37, 0] - label[41, 0]) + np.square(label[37, 1] - label[41, 1])) / bbox_height < 0.02 \
                             or np.sqrt(
                np.square(label[38, 0] - label[40, 0]) + np.square(label[38, 1] - label[40, 1])) / bbox_height < 0.02
            right_eye_close = np.sqrt(
                np.square(label[43, 0] - label[47, 0]) + np.square(label[43, 1] - label[47, 1])) / bbox_height < 0.02 \
                              or np.sqrt(
                np.square(label[44, 0] - label[46, 0]) + np.square(label[44, 1] - label[46, 1])) / bbox_height < 0.02
            if left_eye_close or right_eye_close:
                for i in range(10):
                    res_anns.append(ann)
            ####half face
            if np.sqrt(
                    np.square(label[36, 0] - label[45, 0]) + np.square(label[36, 1] - label[45, 1])) / bbox_width < 0.5:
                for i in range(20):
                    res_anns.append(ann)

            if np.sqrt(np.square(label[62, 0] - label[66, 0]) + np.square(
                    label[62, 1] - label[66, 1])) / bbox_height > 0.15:
                for i in range(20):
                    res_anns.append(ann)
            ##########eyes diff aug
            if left_eye_close and not right_eye_close:
                for i in range(40):
                    res_anns.append(ann)
                lar_count += 1
            if not left_eye_close and right_eye_close:
                for i in range(40):
                    res_anns.append(ann)
                lar_count += 1
        logger.info('the dataset with big mouth open  %d images' % (lar_count))
        random.shuffle(res_anns)
        logger.info('befor balance the dataset contains %d images' % (len(anns)))
        logger.info('after balanced the datasets contains %d samples' % (len(res_anns)))
        return res_anns

    def parse_file(self,im_root_path,ann_file):
        '''
        :return:
        '''
        logger.info("[x] Get dataset from {}".format(im_root_path))

        ann_info = data_info(im_root_path, ann_file)
        all_samples = ann_info.get_all_sample()
        #balanced_samples = self.balance(all_samples)
        return all_samples

    def augmentationCropImage(self,img, bbox, joints=None, is_training=True):

        bbox = np.array(bbox).reshape(4, ).astype(np.float32)
        add = max(img.shape[0], img.shape[1])

        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=cfg.DATA.PIXEL_MEAN)

        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
        bbox += add
        objcenter += add

        joints[:, :2] += add

        gt_width = (bbox[2] - bbox[0])
        gt_height = (bbox[3] - bbox[1])

        crop_width_half = gt_width * (1 + cfg.DATA.base_extend_range[0] * 2) // 2
        crop_height_half = gt_height * (1 + cfg.DATA.base_extend_range[1] * 2) // 2

        if is_training:
            min_x = int(objcenter[0] - crop_width_half + \
                        random.uniform(-cfg.DATA.base_extend_range[0], cfg.DATA.base_extend_range[0]) * gt_width)
            max_x = int(objcenter[0] + crop_width_half + \
                        random.uniform(-cfg.DATA.base_extend_range[0], cfg.DATA.base_extend_range[0]) * gt_width)
            min_y = int(objcenter[1] - crop_height_half + \
                        random.uniform(-cfg.DATA.base_extend_range[1], cfg.DATA.base_extend_range[1]) * gt_height)
            max_y = int(objcenter[1] + crop_height_half + \
                        random.uniform(-cfg.DATA.base_extend_range[1], cfg.DATA.base_extend_range[1]) * gt_height)
        else:
            min_x = int(objcenter[0] - crop_width_half)
            max_x = int(objcenter[0] + crop_width_half)
            min_y = int(objcenter[1] - crop_height_half)
            max_y = int(objcenter[1] + crop_height_half)

        joints[:, 0] = joints[:, 0] - min_x
        joints[:, 1] = joints[:, 1] - min_y

        img = bimg[min_y:max_y, min_x:max_x, :]

        crop_image_height, crop_image_width, _ = img.shape
        joints[:, 0] = joints[:, 0] / crop_image_width
        joints[:, 1] = joints[:, 1] / crop_image_height


        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
                          cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)

        img = cv2.resize(img, (cfg.MODEL.win, cfg.MODEL.hin),interpolation=interp_method)

        joints[:, 0] = joints[:, 0] * cfg.MODEL.win
        joints[:, 1] = joints[:, 1] * cfg.MODEL.hin
        return img, joints
    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here
        fname, ann = dp
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = np.array([ann.split(' ')], dtype=np.float).reshape((-1, 2))

        bbox = np.array([np.min(label[:, 0]), np.min(label[:, 1]), np.max(label[:, 0]), np.max(label[:, 1])])
        bbox_height = bbox[3] - bbox[1]
        bbox_width = bbox[2] - bbox[0]



        crop_image, label = self.augmentationCropImage(image, bbox, label, is_training)

        if is_training:

            if random.uniform(0, 1) > 0.5:
                crop_image, label = Mirror(crop_image, label=label, symmetry=cfg.DATA.symmetry)
            if random.uniform(0, 1) > 0.5:
                strength = random.uniform(0, 50)
                crop_image, label = Affine_aug(crop_image, strength=strength, label=label)
            if random.uniform(0, 1) > 0.5:
                crop_image = Padding_aug(crop_image, 0.3)
            if random.uniform(0, 1) > 0.0:
                angle = random.uniform(-45, 45)
                crop_image, label = Rotate_aug(crop_image, label=label, angle=angle)
            #
            #
            if random.uniform(0, 1) > 0.5:
                crop_image=self.color_augmentor(crop_image)
        #######head pose
        reprojectdst, euler_angle = get_head_pose(label, crop_image)
        PRY = euler_angle.reshape([-1]).astype(np.float32) / 90.

        ######cla_label
        cla_label = np.zeros([3])
        if np.sqrt(np.square(label[37, 0] - label[41, 0]) + np.square(label[37, 1] - label[41, 1])) / bbox_height < 0.02 \
                or np.sqrt(
            np.square(label[38, 0] - label[40, 0]) + np.square(label[38, 1] - label[40, 1])) / bbox_height < 0.02:
            cla_label[0] = 1
        if np.sqrt(np.square(label[43, 0] - label[47, 0]) + np.square(label[43, 1] - label[47, 1])) / bbox_height < 0.02 \
                or np.sqrt(
            np.square(label[44, 0] - label[46, 0]) + np.square(label[44, 1] - label[46, 1])) / bbox_height < 0.02:
            cla_label[1] = 1
        if np.sqrt(np.square(label[61, 0] - label[67, 0]) + np.square(label[61, 1] - label[67, 1])) / bbox_height < 0.02 \
                or np.sqrt(
            np.square(label[62, 0] - label[66, 0]) + np.square(label[62, 1] - label[66, 1])) / bbox_height < 0.02 \
                or np.sqrt(
            np.square(label[63, 0] - label[65, 0]) + np.square(label[63, 1] - label[65, 1])) / bbox_height < 0.02:
            cla_label[2] = 1

        crop_image_height, crop_image_width, _ = crop_image.shape

        label = label.astype(np.float32)


        label[:, 0] = label[:, 0] / crop_image_width
        label[:, 1] = label[:, 1] / crop_image_height

        crop_image = crop_image.astype(np.float32)

        label = label.reshape([-1]).astype(np.float32)
        cla_label = cla_label.astype(np.float32)
        label = np.concatenate([label, PRY, cla_label], axis=0)

        return crop_image, label,
