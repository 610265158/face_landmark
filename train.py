
from lib.helper.logger import logger
from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import FaceKeypointDataIter
from lib.core.model.simpleface import SimpleFace

import tensorflow as tf

from train_config import config as cfg
import setproctitle
setproctitle.setproctitle("face*_*_")














def main():

    epochs=cfg.TRAIN.epoch
    batch_size=cfg.TRAIN.batch_size

    enable_function=False

    devices = ['/device:GPU:{}'.format(i) for i in range(cfg.TRAIN.num_gpu)]


    strategy = tf.distribute.MirroredStrategy(devices)
    with strategy.scope():
        model=SimpleFace()
    trainer = Train(epochs, enable_function, model, batch_size, strategy)




    train_ds = FaceKeypointDataIter(cfg.DATA.root_path, cfg.DATA.train_txt_path, True)
    test_ds = FaceKeypointDataIter(cfg.DATA.root_path, cfg.DATA.val_txt_path, False)


    train_dataset=tf.data.Dataset.from_generator(train_ds,
                                                 output_types=(tf.float32,tf.float32),
                                                 output_shapes=([None,None,None],[cfg.MODEL.out_channel]))
    test_dataset = tf.data.Dataset.from_generator(test_ds,
                                                  output_types=(tf.float32,tf.float32),
                                                  output_shapes=([None,None,None],[cfg.MODEL.out_channel]))



    train_dataset = train_dataset.batch(128).prefetch(100)
    test_dataset = test_dataset.batch(128).prefetch(100)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
    # for X, y in train_dataset:
    #     print(X.shape)
    #     print(y)

    trainer.custom_loop(train_dist_dataset,
                        test_dist_dataset,
                        strategy)


if __name__=='__main__':
    main()