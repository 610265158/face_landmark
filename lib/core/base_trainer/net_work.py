#-*-coding:utf-8-*-


import tensorflow as tf


import time
import numpy as np
import cv2
import os

from train_config import config as cfg
#from lib.dataset.dataietr import DataIter

from lib.core.model.simpleface import SimpleFace
from lib.core.model.simpleface import calculate_loss
from lib.helper.logger import logger


class Train(object):
  """Train class.
  Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
    model: Densenet model.
    batch_size: Batch size.
    strategy: Distribution strategy in use.
  """

  def __init__(self, epochs, enable_function, model, batch_size, strategy):
    self.epochs = epochs
    self.batch_size = batch_size
    self.enable_function = enable_function
    self.strategy = strategy


    self.loss_object = self.loss_obj


    if 'Adam' in cfg.TRAIN.opt:
      self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.TRAIN.lr_value_every_step[0])
    else:
      self.optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.TRAIN.lr_value_every_step[0],momentum=0.9)

    self.model = model




    ###control vars
    self.iter_num=0

    self.lr_decay_every_epoch =cfg.TRAIN.lr_decay_every_epoch
    self.lr_val_every_epoch = cfg.TRAIN.lr_value_every_step


    ##

  def loss_obj(self,label,predictions):
      loss=calculate_loss(predictions,label)
      return loss




  def decay(self, epoch):
    if epoch < self.lr_decay_every_epoch[0]:
      return self.lr_val_every_epoch[0]
    if epoch >= self.lr_decay_every_epoch[0] and epoch < self.lr_decay_every_epoch[1]:
      return self.lr_val_every_epoch[1]
    if epoch >= self.lr_decay_every_epoch[1] and epoch < self.lr_decay_every_epoch[2]:
      return self.lr_val_every_epoch[2]
    if epoch >= self.lr_decay_every_epoch[2]:
      return self.lr_val_every_epoch[3]

  def compute_loss(self, label, predictions):
    loss = tf.reduce_sum(self.loss_object(label, predictions))
    loss += (sum(self.model.losses) * 1. / self.strategy.num_replicas_in_sync)
    return loss

  def train_step(self, inputs):
    """One train step.
    Args:
      inputs: one batch input.
    Returns:
      loss: Scaled loss.
    """

    image, label = inputs
    with tf.GradientTape() as tape:
      predictions = self.model(image, training=True)

      # img = np.array(image[0], dtype=np.uint8)
      # landmark = np.array(predictions[0][0:136]).reshape([-1, 2])
      #
      # for _index in range(landmark.shape[0]):
      #   x_y = landmark[_index]
      #
      #   cv2.circle(img, center=(int(x_y[0] * 160),
      #                           int(x_y[1] * 160)),
      #              color=(255, 122, 122), radius=1, thickness=2)
      #
      # cv2.imwrite('tmp.jpg',img)

      loss = self.compute_loss(label, predictions)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    gradients = [(tf.clip_by_value(grad, -5.0, 5.0))
                 for grad in gradients]
    self.optimizer.apply_gradients(zip(gradients,
                                       self.model.trainable_variables))

    return loss

  def test_step(self, inputs):
    """One test step.
    Args:
      inputs: one batch input.
    """

    image, label = inputs

    predictions = self.model(image,training=False)



    ### check process
    img = np.array(image[0], dtype=np.uint8)
    landmark = np.array(predictions[0][0:136]).reshape([-1, 2])

    for _index in range(landmark.shape[0]):
      x_y = landmark[_index]

      cv2.circle(img, center=(int(x_y[0] * 160),
                              int(x_y[1] * 160)),
                 color=(255, 122, 122), radius=1, thickness=2)

    cv2.imwrite('valtmp.jpg', img)
    ### check process

    unscaled_test_loss = self.compute_loss(label, predictions)

    return unscaled_test_loss


  def custom_loop(self, train_dist_dataset, test_dist_dataset,
                  strategy):
    """Custom training and testing loop.
    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.
    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def distributed_train_epoch(ds):
      total_loss = 0.0
      num_train_batches = 0.0
      for one_batch in ds:

        start=time.time()
        per_replica_loss = strategy.experimental_run_v2(
            self.train_step, args=(one_batch,))
        current_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        total_loss += current_loss
        num_train_batches += 1
        self.iter_num+=1
        time_cost_per_batch=time.time()-start

        images_per_sec=cfg.TRAIN.batch_size/time_cost_per_batch
        if self.iter_num%cfg.TRAIN.log_interval==0:
          logger.info('iter_num: %d, '
                      'loss_value: %.6f,  '
                      'speed: %d images/sec ' \
                      % (self.iter_num, current_loss,images_per_sec))

      return total_loss, num_train_batches

    def distributed_test_epoch(ds):
      total_loss=0.
      num_test_batches = 0.0
      for one_batch in ds:
        per_replica_loss=strategy.experimental_run_v2(
            self.test_step, args=(one_batch,))

        current_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        total_loss+=current_loss
        num_test_batches += 1
      return total_loss, num_test_batches

    if self.enable_function:
      distributed_train_epoch = tf.function(distributed_train_epoch)
      distributed_test_epoch = tf.function(distributed_test_epoch)

    for epoch in range(self.epochs):

      start=time.time()
      self.optimizer.learning_rate = self.decay(epoch)

      train_total_loss, num_train_batches = distributed_train_epoch(
          train_dist_dataset)
      test_total_loss, num_test_batches = distributed_test_epoch(
          test_dist_dataset)



      time_consume_per_epoch=time.time()-start
      training_massage = 'Epoch: %d, ' \
                         'Train Loss: %.6f, ' \
                         'Test Loss: %.6f '\
                         'Time consume: %.2f'%(epoch,
                                               train_total_loss / num_train_batches,
                                               test_total_loss / num_test_batches,
                                               time_consume_per_epoch)

      logger.info(training_massage)


      #### save the model every end of epoch
      current_model_saved_name=os.path.join(cfg.MODEL.model_path,
                                            'epoch_%d_val_loss%.6f_keras.h5'%(epoch,test_total_loss / num_test_batches))

      if not os.access(cfg.MODEL.model_path,os.F_OK):
        os.mkdir(cfg.MODEL.model_path)
      self.model.save_weights(current_model_saved_name)



    return (train_total_loss / num_train_batches,
            test_total_loss / num_test_batches)





