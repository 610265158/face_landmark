from lib.helper.logger import logger
from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import DataIter
from lib.core.model.shufflenet.simpleface import SimpleFace as SimpleFace_shufflenet
from lib.core.model.mobilenet.simpleface import SimpleFace as SimpleFace_mobilenet

import tensorflow as tf
import cv2
import numpy as np

from train_config import config as cfg
import setproctitle

logger.info('The trainer start')

setproctitle.setproctitle("face*_*_")


def main():

    epochs=cfg.TRAIN.epoch
    batch_size=cfg.TRAIN.batch_size

    enable_function=False

    ### set gpu memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    devices = ['/device:GPU:{}'.format(i) for i in range(cfg.TRAIN.num_gpu)]
    strategy = tf.distribute.MirroredStrategy(devices)

    ##build model
    with strategy.scope():
        if 'ShuffleNet' in cfg.MODEL.net_structure:
            model=SimpleFace_shufflenet()
        else:
            model = SimpleFace_mobilenet()
        ##run a time to build
        image = np.zeros(shape=(1, 160, 160, 3), dtype=np.float32)
        model(image)

    ###recover weights
    if cfg.MODEL.pretrained_model is not None:
        model.load_weights(cfg.MODEL.pretrained_model)


    ###build trainer
    trainer = Train(epochs, enable_function, model, batch_size, strategy)



    ###build dataset
    train_ds = DataIter(cfg.DATA.root_path, cfg.DATA.train_txt_path, True)
    test_ds = DataIter(cfg.DATA.root_path, cfg.DATA.val_txt_path, False)


    train_dataset=tf.data.Dataset.from_generator(train_ds,
                                                 output_types=(tf.float32,tf.float32),
                                                 output_shapes=([None,None,None,None],[None,cfg.MODEL.out_channel]))
    test_dataset = tf.data.Dataset.from_generator(test_ds,
                                                  output_types=(tf.float32,tf.float32),
                                                  output_shapes=([None,None,None,None],[None,cfg.MODEL.out_channel]))

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    if cfg.TRAIN.vis:
        for images, labels in train_dist_dataset:

            for i in range(images.shape[0]):
                example_image=np.array(images[i],dtype=np.uint8)
                example_label=np.array(labels[i])

                landmark = example_label[0:136].reshape([-1, 2])
                _h, _w, _ = example_image.shape

                print(landmark.shape)
                for _index in range(landmark.shape[0]):
                    x_y = landmark[_index]
                    cv2.circle(example_image, center=(int(x_y[0] * _w), int(x_y[1] * _w)), color=(255, 255, 255),
                               radius=1, thickness=-1)

                cv2.imshow('example',example_image)
                cv2.waitKey(0)



    ### train
    trainer.custom_loop(train_dist_dataset,
                        test_dist_dataset,
                        strategy)

if __name__=='__main__':
    main()