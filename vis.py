from lib.dataset.dataietr import FaceKeypointDataIter
from train_config import config
from lib.core.api.keypoint import Keypoints
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from train_config import config as cfg
cfg.TRAIN.batch_size=1

val_ds = FaceKeypointDataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,False)

face=Keypoints('./model/keypoints.pb')


for one_ele,_, in val_ds:
    print(_)

    img_show=np.array(one_ele)
    res=face.simple_run(one_ele)
    #print(res)
    res=res[0][:136].reshape((-1,2))
    img_show=img_show.astype(np.uint8)

    img_show=cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

    for _index in range(res.shape[0]):
        x_y = res[_index]
        cv2.circle(img_show, center=(int(x_y[0] * config.MODEL.hin),
                                     int(x_y[1] * config.MODEL.win)),
                   color=(255, 122, 122), radius=1, thickness=2)

    cv2.imshow('tmp',img_show)
    cv2.waitKey(0)
