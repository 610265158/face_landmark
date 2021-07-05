





import cv2
import numpy as np


import setproctitle

from train_config import config as cfg
from lib.helper.logger import logger
from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import FaceKeypointDataIter
from lib.core.model.face_model import Net
logger.info('The trainer start')


from torch.utils.data import Dataset, DataLoader
setproctitle.setproctitle("face*_*_")


def main():
    ###build dataset
    train_generator = FaceKeypointDataIter(cfg.DATA.root_path, cfg.DATA.train_txt_path, training_flag=True)
    train_ds = DataLoader(train_generator,
                          cfg.TRAIN.batch_size,
                          num_workers=cfg.TRAIN.process_num,
                          shuffle=True)
    val_generator = FaceKeypointDataIter(cfg.DATA.root_path, cfg.DATA.val_txt_path, training_flag=False)
    val_ds = DataLoader(val_generator,
                          cfg.TRAIN.batch_size,
                          num_workers=cfg.TRAIN.process_num,
                          shuffle=False)


    #build model
    model = Net(num_classes=cfg.MODEL.out_channel)

    ###build trainer
    trainer = Train( model,train_ds=train_ds,val_ds=val_ds)


    if cfg.TRAIN.vis:
        for images, labels in train_ds:



            for i in range(images.shape[0]):

                example_image=images[i].numpy().transpose([1,2,0]).copy()

                example_label=labels[i].numpy()

                landmark = example_label[0:136].reshape([-1, 2])
                _h, _w, _ = example_image.shape

                for _index in range(landmark.shape[0]):
                    x_y = landmark[_index]
                    cv2.circle(example_image, center=(int(x_y[0] * _w), int(x_y[1] * _w)), color=(255, 255, 255),
                               radius=1, thickness=-1)

                cv2.imshow('example',example_image)
                cv2.waitKey(0)



    ### train
    trainer.custom_loop()

if __name__=='__main__':
    main()