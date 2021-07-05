from torch.utils.data import DataLoader

from lib.dataset.dataietr import FaceKeypointDataIter
from train_config import config


import torch
import time
import argparse


import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
from train_config import config as cfg
cfg.TRAIN.batch_size=1

ds = FaceKeypointDataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,False)
ds = DataLoader(ds,
                         1,
                          num_workers=cfg.TRAIN.process_num,
                          shuffle=False)
from lib.core.model.face_model import Net
face=Net(num_classes=cfg.MODEL.out_channel)

face
def vis(model):

    ###build model


    state_dict = torch.load(model, map_location=torch.device('cpu'))
    face.load_state_dict(state_dict, strict=False)

    face.eval()
    for images, labels in ds:


        img_show = images.numpy()
        print(img_show.shape)
        img_show=np.transpose(img_show[0],axes=[1,2,0])

        images=images.to('cpu').float()

        start=time.time()
        res=face(images)
        res=res.detach().numpy()
        print(res)
        print('xxxx',time.time()-start)
        #print(res)

        img_show=img_show.astype(np.uint8)

        img_show=cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        landmark = np.array(res[0][0:136]).reshape([-1, 2])

        for _index in range(landmark.shape[0]):
            x_y = landmark[_index]
            #print(x_y)
            cv2.circle(img_show, center=(int(x_y[0] * 128),
                                         int(x_y[1] * 128)),
                       color=(255, 122, 122), radius=1, thickness=2)

        cv2.imshow('tmp',img_show)
        cv2.waitKey(0)


def load_checkpoint(net, checkpoint):
    # from collections import OrderedDict
    #
    # temp = OrderedDict()
    # if 'state_dict' in checkpoint:
    #     checkpoint = dict(checkpoint['state_dict'])
    # for k in checkpoint:
    #     k2 = 'module.'+k if not k.startswith('module.') else k
    #     temp[k2] = checkpoint[k]

    net.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu')), strict=True)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Start train.')

    parser.add_argument('--model', dest='model', type=str, default=None, \
                        help='the model to use')

    args = parser.parse_args()


    vis(args.model)




