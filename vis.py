from data.utils import get_train_data_list,_data_aug_fn
from train_config import config
from api.keypoint import Keypoints
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2

val_data_list=get_train_data_list(config.DATA.root_path,config.DATA.val_txt_path)

face=Keypoints('./model/landmark.pb')
for one_ele in val_data_list:

    img=cv2.imread(one_ele[0])
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop_img,_label,_heatmaps=_data_aug_fn(img,one_ele[1],False)

    img_show=crop_img.copy()

    res=face.simple_run(crop_img)
    print(res)
    res=res[0][:136].reshape((-1,2))
    img_show=img_show.astype(np.uint8)


    for _index in range(res.shape[0]):
        x_y = res[_index]
        cv2.circle(img_show, center=(int(x_y[0] * config.MODEL.hin),
                                     int(x_y[1] * config.MODEL.win)),
                   color=(255, 122, 122), radius=1, thickness=2)

    cv2.imshow('tmp',img_show)
    cv2.waitKey(0)