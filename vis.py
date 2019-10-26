from lib.dataset.dataietr import FaceKeypointDataIter
from train_config import config
import tensorflow as tf
import time
import argparse


import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from train_config import config as cfg
cfg.TRAIN.batch_size=1

ds = FaceKeypointDataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,False)
train_dataset = tf.data.Dataset.from_generator(ds,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, None, None], [cfg.MODEL.out_channel]))


def vis(model):

    ###build model
    face = tf.saved_model.load(model)

    for images, labels in train_dataset:
        img_show = np.array(images)

        images=np.expand_dims(images,axis=0)
        start=time.time()
        res=face.inference(images)
        print('xxxx',time.time()-start)
        #print(res)

        img_show=img_show.astype(np.uint8)

        img_show=cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        landmark = np.array(res['landmark'][0]).reshape([-1, 2])

        for _index in range(landmark.shape[0]):
            x_y = landmark[_index]
            #print(x_y)
            cv2.circle(img_show, center=(int(x_y[0] * 160),
                                         int(x_y[1] * 160)),
                       color=(255, 122, 122), radius=1, thickness=2)

        cv2.imshow('tmp',img_show)
        cv2.waitKey(0)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Start train.')

    parser.add_argument('--model', dest='model', type=str, default=None, \
                        help='the model to use')

    args = parser.parse_args()

    vis(args.model)




