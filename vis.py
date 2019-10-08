from lib.dataset.dataietr import FaceKeypointDataIter
from train_config import config
import tensorflow as tf
import time


from lib.core.model.simpleface import SimpleFace
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from train_config import config as cfg
cfg.TRAIN.batch_size=1

ds = FaceKeypointDataIter(cfg.DATA.root_path,cfg.DATA.train_txt_path,True)
train_dataset = tf.data.Dataset.from_generator(ds,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, None, None], [cfg.MODEL.out_channel]))


saved_model_path='./model/epoch_3_val_loss91.378777_keras.h5'

###build model
face=SimpleFace()
image=np.zeros(shape=(1,160,160,3),dtype=np.float32)
face(image)

#face.load_weights(saved_model_path)

# saved_model_path='./model/epoch_4_val_loss108.853745'
# face = tf.keras.models.load_model(saved_model_path)




for images, labels in train_dataset:
    img_show = np.array(images)

    images=np.expand_dims(images,axis=0)
    start=time.time()
    res=face(images,training=False)
    print('xxxx',time.time()-start)
    #print(res)

    img_show=img_show.astype(np.uint8)

    img_show=cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

    landmark = np.array(res[0][0:136]).reshape([-1, 2])

    for _index in range(landmark.shape[0]):
        x_y = landmark[_index]
        #print(x_y)
        cv2.circle(img_show, center=(int(x_y[0] * 160),
                                     int(x_y[1] * 160)),
                   color=(255, 122, 122), radius=1, thickness=2)

    cv2.imshow('tmp',img_show)
    cv2.waitKey(0)
