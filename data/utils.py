#-*-coding:utf-8-*-

import pickle
import tensorflow as tf
import numpy as np
import cv2
import random
from functools import partial
import copy

from data.headpose import get_head_pose
from helper.logger import logger
from data.datainfo import data_info
from data.augmentor.augmentation import Pixel_jitter,\
                                        Rotate_aug,\
                                        Affine_aug,\
                                        Mirror,\
                                        Random_contrast,\
                                        Random_brightness,\
                                        Padding_aug,\
                                        Blur_aug,\
                                        produce_heat_maps

from train_config import config as cfg
from tensorpack.dataflow import BatchData, MultiThreadMapData, PrefetchDataZMQ,DataFromList
def balance(anns):
    res_anns=copy.deepcopy(anns)

    lar_count=0
    for ann in anns:
        label=ann[-1]
        label = np.array([label.split(' ')], dtype=np.float).reshape((-1, 2))
        bbox = np.array([np.min(label[:, 0]), np.min(label[:, 1]), np.max(label[:, 0]), np.max(label[:, 1])])
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        if bbox_width<30 or bbox_height<30:
            res_anns.remove(ann)
        

        left_eye_close=np.sqrt(np.square(label[37,0]-label[41,0])+np.square(label[37,1]-label[41,1]))/bbox_height<0.02 \
            or np.sqrt(np.square(label[38, 0] - label[40, 0]) + np.square(label[38, 1] - label[40, 1])) / bbox_height < 0.02
        right_eye_close=np.sqrt(np.square(label[43,0]-label[47,0])+np.square(label[43,1]-label[47,1]))/bbox_height<0.02 \
            or np.sqrt(np.square(label[44, 0] - label[46, 0]) + np.square(label[44, 1] - label[46, 1])) / bbox_height < 0.02
        if left_eye_close or right_eye_close:
            for i in range(10):
                res_anns.append(ann)
        ####half face
        if np.sqrt(np.square(label[36,0]-label[45,0])+np.square(label[36,1]-label[45,1]))/bbox_width<0.5:
            for i in range(20):
                res_anns.append(ann)
    
        if np.sqrt(np.square(label[62,0]-label[66,0])+np.square(label[62,1]-label[66,1]))/bbox_height>0.15:
            for i in range(20):
                res_anns.append(ann)
        ##########eyes diff aug
        if left_eye_close and not right_eye_close:
            for i in range(40):
                res_anns.append(ann)
            lar_count+=1
        if not left_eye_close and right_eye_close:
            for i in range(40):
                res_anns.append(ann)
            lar_count+=1
    logger.info('the dataset with big mouth open  %d images' % (lar_count))
    random.shuffle(res_anns)
    logger.info('befor balance the dataset contains %d images' % (len(anns)))
    logger.info('after balanced the datasets contains %d samples' % (len(res_anns)))
    return res_anns
def get_train_data_list(im_root_path, ann_txt):
    """
    train_im_path : image folder name
    train_ann_path : coco json file name
    """
    logger.info("[x] Get data from {}".format(im_root_path))
    # data = PoseInfo(im_path, ann_path, False)
    data = data_info(im_root_path, ann_txt)
    all_samples=data.get_all_sample()

    balanced_samples=balance(all_samples)
    print(len(balanced_samples))
    print(len(all_samples))
    return balanced_samples
def get_data_set(root_path,ana_path):
    data_list=get_train_data_list(root_path,ana_path)
    dataset= DataFromList(data_list, shuffle=True)
    return dataset

def augmentationCropImage(img, bbox, joints=None,is_training=True):
    height, width = cfg.MODEL.hin, cfg.MODEL.win
    bbox = np.array(bbox).reshape(4, ).astype(np.float32)
    add = max(img.shape[0], img.shape[1])

    bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=cfg.DATA.PIXEL_MEAN)

    objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
    bbox += add
    objcenter += add
    if is_training:
        joints[:, :2] += add
        inds = np.where(joints[:, -1] == 0)
        joints[inds, :2] = -1000000  # avoid influencing by data processing
    crop_width = (bbox[2] - bbox[0]) * (1 + cfg.DATA.base_extend_range[0] * 2)
    crop_height = (bbox[3] - bbox[1]) * (1 + cfg.DATA.base_extend_range[1] * 2)
    if is_training:
        crop_width = crop_width * (1 + 0.25)
        crop_height = crop_height * (1 + 0.25)
    if crop_height / height > crop_width / width:
        crop_size = crop_height
        min_shape = height
    else:
        crop_size = crop_width
        min_shape = width

    crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
    crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

    min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
    max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
    min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
    max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

    x_ratio = float(width) / (max_x - min_x)
    y_ratio = float(height) / (max_y - min_y)

    if is_training:
        joints[:, 0] = joints[:, 0] - min_x
        joints[:, 1] = joints[:, 1] - min_y

        joints[:, 0] *= x_ratio
        joints[:, 1] *= y_ratio

    img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))
    details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add]).astype(np.float)


    height, width = img.shape[0], img.shape[1]
    center = (width / 2., height / 2.)
    n = joints.shape[0]

    affrat = random.uniform(cfg.DATA.scale_factor[0], cfg.DATA.scale_factor[1])

    halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
    halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)
    img = cv2.resize(img[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                     int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (width, height),interpolation=interp_method)

    for i in range(n):
        joints[i][0] = (joints[i][0] - center[0]) / halfl_w * (width - center[0]) + center[0]
        joints[i][1] = (joints[i][1] - center[1]) / halfl_h * (height - center[1]) + center[1]

    if is_training:
        return img, joints, details
    else:
        return img, details


def _data_aug_fn(image, ground_truth,is_training=True):
    """Data augmentation function."""
    ####customed here

    label = np.array([ground_truth.split(' ')],dtype=np.float).reshape((-1,2))


    bbox=np.array([np.min(label[:,0]),np.min(label[:,1]),np.max(label[:,0]),np.max(label[:,1])])
    bbox_height=bbox[3]-bbox[1]
    bbox_width = bbox[2] - bbox[0]

    crop_image, label, _=augmentationCropImage(image,bbox,label,True)


    if is_training:

        ###affine_aug
        if random.uniform(0, 1) > 0.5:
            strength = random.uniform(0, 50)
            crop_image, label = Affine_aug(crop_image, strength=strength, label=label)
        if random.uniform(0, 1) > 0.5:
            crop_image = Padding_aug(crop_image, 0.4)
        if random.uniform(0,1)>0.0:
            angle=random.uniform(-45,45)
            crop_image,label=Rotate_aug(crop_image,label=label,angle=angle)


        if random.uniform(0, 1) > 0.5:
            crop_image = Random_brightness(crop_image, bright_shrink=30)
        if random.uniform(0, 1) > 0.5:
            crop_image = Random_contrast(crop_image, [0.5, 1.5])
        if random.uniform(0, 1) > 0.5:
            crop_image = Pixel_jitter(crop_image,15)

        ###mirror
        if random.uniform(0,1)>0.5:
            crop_image,label=Mirror(crop_image,label=label,symmetry=cfg.DATA.symmetry)
        

        if random.uniform(0,1)>0.5:
            a=[3,5,7,9]
            k=random.sample(a, 1)[0]
            crop_image=Blur_aug(crop_image,ksize=(k,k))

    #######head pose
    reprojectdst, euler_angle = get_head_pose(label, crop_image)
    PRY=euler_angle.reshape([-1]).astype(np.float32)/90.

    ######cla_label
    cla_label=np.zeros([3])
    if np.sqrt(np.square(label[37,0]-label[41,0])+np.square(label[37,1]-label[41,1]))/bbox_height<0.02 \
        or np.sqrt(np.square(label[38, 0] - label[40, 0]) + np.square(label[38, 1] - label[40, 1])) / bbox_height < 0.02:
        cla_label[0]=1
    if np.sqrt(np.square(label[43,0]-label[47,0])+np.square(label[43,1]-label[47,1]))/bbox_height<0.02 \
        or np.sqrt(np.square(label[44, 0] - label[46, 0]) + np.square(label[44, 1] - label[46, 1])) / bbox_height < 0.02:
        cla_label[1]=1
    if np.sqrt(np.square(label[61,0]-label[67,0])+np.square(label[61,1]-label[67,1]))/bbox_height<0.02 \
        or np.sqrt(np.square(label[62, 0] - label[66, 0]) + np.square(label[62, 1] - label[66, 1])) / bbox_height < 0.02 \
        or np.sqrt(np.square(label[63, 0] - label[65, 0]) + np.square(label[63, 1] - label[65, 1])) / bbox_height < 0.02:
        cla_label[2]=1



    crop_image_height, crop_image_width, _ = crop_image.shape



    label=label.astype(np.float32)

    label[:, 0] = label[:, 0]/crop_image_width
    label[:, 1] =  label[:, 1]/crop_image_height


    crop_image = crop_image.astype(np.float32)

    label = label.reshape([-1]).astype(np.float32)
    cla_label = cla_label.astype(np.float32)
    label=np.concatenate([label,PRY,cla_label],axis=0)

    return crop_image, label


def _map_fn(dp,is_training=True):


    fname, annos = dp
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image,label=_data_aug_fn(image,annos,is_training)
    return image, label




