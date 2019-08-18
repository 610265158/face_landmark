

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()
#### below are params for dataiter
config.TRAIN.thread_num = 5
config.TRAIN.process_num = 2
config.TRAIN.buffer_size = 500
config.TRAIN.prefetch_size = 500
############


config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 64
config.TRAIN.save_interval = 5000
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 2000
config.TRAIN.train_set_size=972930  ###########u need be sure
config.TRAIN.val_set_size=107115###50562
config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.lr_value_every_step = [0.001,0.0001,0.00001,0.000001]
config.TRAIN.lr_decay_every_step = [150000,200000,250000]
config.TRAIN.weight_decay_factor = 1.e-4
config.TRAIN.train_val_ratio= 0.9
config.TRAIN.vis=False
config.TRAIN.mix_precision=True          ##use mix precision to speedup


config.MODEL = edict()
config.MODEL.continue_train=False            ##recover from a model
config.MODEL.pruning=False                  ## pruning flag
config.MODEL.model_path = './model/'  # save directory
config.MODEL.hin = 128  # input size during training , 128
config.MODEL.win = 128
config.MODEL.out_channel=136+3+3    # output vector    68 points and 3 classes and 3 headpose params
config.MODEL.net_structure='resnet_v1_50'
config.MODEL.pretrained_model='./resnet_v1_50.ckpt'


config.DATA = edict()

config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
############NOW the model is trained with RGB mode

config.DATA.PIXEL_MEAN = [123.675, 116.28, 103.53]   ###rgb
config.DATA.PIXEL_STD = [58.395, 57.12, 57.375]     ### no use


config.DATA.base_extend_range=[0.2,0.3]              ###extand
config.DATA.scale_factor=[0.7,1.35]                    ###scales
config.DATA.symmetry = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (8, 8),
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
            (31, 35), (32, 34),
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
            (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)]









