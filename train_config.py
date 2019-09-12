

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()
#### below are params for dataiter
config.TRAIN.process_num = 5
config.TRAIN.prefetch_size = 100
############


config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 128
config.TRAIN.save_interval = 5000
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 2000            #### no actual meaning, just keep training,
config.TRAIN.train_set_size=972930  ###########u need be sure
config.TRAIN.val_set_size=107115###50562
config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.lr_value_every_step = [0.001,0.0001,0.00001,0.000001]
config.TRAIN.lr_decay_every_step = [150000,250000,300000]
config.TRAIN.weight_decay_factor = 1.e-5
config.TRAIN.train_val_ratio= 0.9
config.TRAIN.vis=False
config.TRAIN.mix_precision=False          ##use mix precision to speedup
config.TRAIN.opt='Adam'          ##Adam or SGD

config.MODEL = edict()
config.MODEL.continue_train=False            ##recover from a model completly
config.MODEL.model_path = './model/'  # save directory
config.MODEL.hin = 160  # input size during training , 128
config.MODEL.win = 160
config.MODEL.out_channel=136+3+4    # output vector    68 points , 3 headpose ,4 cls params
# config.MODEL.net_structure='resnet_v1_50'         #### resnet_v1_50 or  MobilenetV2 are supported
# config.MODEL.pretrained_model='resnet_v1_50.ckpt'          ###resnet_v1_50.ckpt or mobilenet_v2_1.0_224.ckpt

###
# config.MODEL.net_structure='MobilenetV2'         #### resnet_v1_50 or  MobilenetV2 are supported
# config.MODEL.pretrained_model='mobilenet_v2_0.5_224.ckpt'          ###resnet_v1_50.ckpt or mobilenet_v2_0.5_224.ckpt

config.MODEL.net_structure='ShuffleNetV2'         #### resnet_v1_50 or  MobilenetV2 are supported
config.MODEL.pretrained_model=None          ###resnet_v1_50.ckpt or mobilenet_v2_0.5_224.ckpt

config.DATA = edict()

config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
############NOW the model is trained with RGB mode

config.DATA.PIXEL_MEAN = [123., 116., 103.]   ###rgb
config.DATA.PIXEL_STD = [58., 57., 57.]     ### no use

config.DATA.base_extend_range=[0.2,0.3]              ###extand
config.DATA.scale_factor=[0.7,1.35]                    ###scales
config.DATA.symmetry = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (8, 8),
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
            (31, 35), (32, 34),
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
            (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)]



weights=[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,               #####bouding
                               1.,1.,1.,1.,1.,1.,1.,1.,1.,                                      #####nose
                               1.5,1.5,1.5,1.5,1.5,       1.5,1.5,1.5,1.5,1.5,                                   #####eyebows
                               1.5,1.5,1.5,1.5,1.5,1.5,    1.5,1.5,1.5,1.5,1.5,1.5,                                 #####eyes
                               1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.               #####mouth
                               ]
weights_xy=[[x,x] for x in weights]

config.DATA.weights = np.array(weights_xy,dtype=np.float32).reshape([-1])





config.MODEL.pruning=False               ## pruning flag  add l1 reg to bn/beta, no use for tmp
config.MODEL.pruning_bn_reg=0.00005



