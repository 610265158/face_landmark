#-*-coding:utf-8-*-


import tensorflow as tf


import time
import numpy as np
import cv2


from train_config import config as cfg
#from lib.dataset.dataietr import DataIter

from lib.core.model.simpleface import SimpleFace
from lib.core.model.simpleface import calculate_loss
from lib.helper.logger import logger


# class trainner():
#     def __init__(self):
#         # self.train_ds=DataIter(cfg.DATA.root_path,cfg.DATA.train_txt_path,True)
#         # self.val_ds = DataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,False)
#
#
#
#         self.ite_num=1
#
#
#
#         self.summaries = []
#
#         self.ema_weights = False
#
#         self.strategy = tf.distribute.MirroredStrategy()
#         print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
#
#         with self.strategy.scope():
#
#             self.model=SimpleFace()
#
#
#
#     def get_opt(self):
#
#         with self._graph.as_default():
#             ##set the opt there
#             global_step = tf.get_variable(
#                 'global_step', [],
#                 initializer=tf.constant_initializer(0), dtype=tf.int32, trainable=False)
#
#             # Decay the learning rate
#             lr = tf.train.piecewise_constant(global_step,
#                                              cfg.TRAIN.lr_decay_every_step,
#                                              cfg.TRAIN.lr_value_every_step
#                                              )
#             if cfg.TRAIN.opt=='Adam':
#                 opt = tf.train.AdamOptimizer(lr)
#             else:
#                 opt = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=False)
#
#             if cfg.TRAIN.mix_precision:
#                 opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
#
#             return opt,lr,global_step
#
#     def load_weight(self):
#
#         with self._graph.as_default():
#
#             if cfg.MODEL.continue_train:
#                 #########################restore the params
#                 variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
#                 print(variables_restore)
#
#                 saver2 = tf.train.Saver(variables_restore)
#                 saver2.restore(self._sess, cfg.MODEL.pretrained_model)
#
#             elif cfg.MODEL.pretrained_model is not None  and not cfg.MODEL.pruning:
#                 #########################restore the params
#                 variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=cfg.MODEL.net_structure)
#                 print(variables_restore)
#
#                 saver2 = tf.train.Saver(variables_restore)
#                 saver2.restore(self._sess, cfg.MODEL.pretrained_model)
#             elif cfg.MODEL.pruning:
#                 #########################restore the params
#                 variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
#                 print(variables_restore)
#                 #    print('......................................................')
#                 #    # saver2 = tf.train.Saver(variables_restore)
#                 variables_restore_n = [v for v in variables_restore if
#                                        'output' not in v.name]  # Conv2d_1c_1x1 Bottleneck
#                 # print(variables_restore_n)
#
#                 state_dict=np.load(cfg.MODEL.pretrained_model)
#
#                 state_dict=state_dict['arr_0'][()]
#
#                 for var in variables_restore_n:
#                     var_name=var.name.rsplit(':')[0]
#                     if var_name in state_dict:
#                         logger.info('recover %s from npz file'%var_name)
#                         self._sess.run(tf.assign(var, state_dict[var_name]))
#                     else:
#                         logger.info('the params of %s not in npz file'%var_name)
#             else:
#                 logger.info('no pretrained model, train from sctrach')
#                 # Build an initialization operation to run below.
#
#     def add_summary(self, event):
#         self.summaries.append(event)






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

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    self.test_loss_metric = tf.keras.metrics.Sum(name='test_loss')
    self.model = model


  def loss_obj(self,label,predictions):
      loss=calculate_loss(predictions,label)
      return loss




  def decay(self, epoch):
    if epoch < 150:
      return 0.1
    if epoch >= 150 and epoch < 225:
      return 0.01
    if epoch >= 225:
      return 0.001

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
      loss = self.compute_loss(label, predictions)
      print(loss)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients,
                                       self.model.trainable_variables))

    return loss

  def test_step(self, inputs):
    """One test step.
    Args:
      inputs: one batch input.
    """
    image, label = inputs
    predictions = self.model(image, training=False)

    unscaled_test_loss = self.loss_object(label, predictions) + sum(
        self.model.losses)


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
        per_replica_loss = strategy.experimental_run_v2(
            self.train_step, args=(one_batch,))
        total_loss += strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        num_train_batches += 1



        print(per_replica_loss)
      return total_loss, num_train_batches

    def distributed_test_epoch(ds):
      num_test_batches = 0.0
      for one_batch in ds:
        strategy.experimental_run_v2(
            self.test_step, args=(one_batch,))
        num_test_batches += 1
      return self.test_loss_metric.result(), num_test_batches

    if self.enable_function:
      distributed_train_epoch = tf.function(distributed_train_epoch)
      distributed_test_epoch = tf.function(distributed_test_epoch)

    for epoch in range(self.epochs):
      self.optimizer.learning_rate = self.decay(epoch)

      train_total_loss, num_train_batches = distributed_train_epoch(
          train_dist_dataset)
      test_total_loss, num_test_batches = distributed_test_epoch(
          test_dist_dataset)

      template = ('Epoch: {}, Train Loss: {}, '
                  'Test Loss: {}')

      print(
          template.format(epoch,
                          train_total_loss / num_train_batches,
                          test_total_loss / num_test_batches))


    return (train_total_loss / num_train_batches,
            test_total_loss / num_test_batches)





