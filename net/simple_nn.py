#-*-coding:utf-8-*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
from train_config import config as cfg

import numpy as np

def shufflenet_arg_scope(weight_decay=0.00001,
                     batch_norm_decay=0.99,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     use_batch_norm=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
  """Defines the default ResNet arg scope.
  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': batch_norm_updates_collections,
      'fused': True,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d,slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc



######channel shuffle unit
def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]

        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]

        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


#### a depwise conv unit
def basic_unit(x):
    in_channels = x.shape[3].value

    x = slim.conv2d(x, in_channels, [1, 1], stride=1, activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm, scope='conv1x1_before')

    x = slim.separable_conv2d(x, in_channels, [3, 3], stride=1, activation_fn=None,
                              normalizer_fn=slim.batch_norm, scope='depthwise', depth_multiplier=1)

    x = slim.conv2d(x, in_channels, [1, 1], stride=1, activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm, scope='conv1x1_after')
    return x
#### a depwise conv unit with downsampling
def basic_unit_with_downsampling(x,out_channels=None):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, in_channels, [1, 1], stride=1, activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm, scope='conv1x1_before')

    y = slim.separable_conv2d(y, in_channels, [3, 3], stride=2, activation_fn=None,
                    normalizer_fn=slim.batch_norm, scope='depthwise',depth_multiplier=1)

    y = slim.conv2d(y, out_channels//2, [1, 1], stride=1, activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm, scope='conv1x1_after')


    with tf.variable_scope('second_branch'):
        x = slim.separable_conv2d(x, in_channels, [3, 3], stride=2, activation_fn=None,
                                  normalizer_fn=slim.batch_norm, scope='depthwise',depth_multiplier=1)
        x = slim.conv2d(x, out_channels // 2, [1, 1], stride=1, activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm, scope='conv1x1_after')

    x = tf.concat([x, y], axis=3)
    return x

### a simple conv unit
def basic_unit_plain(x):
    in_channels = x.shape[3].value

    x = slim.conv2d(x, in_channels, [3, 3], stride=1, activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm, scope='conv1x1_before')

    return x

### a simple conv unit with downsampling
def basic_unit_with_downsampling_plain(x,out_channels=None):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, out_channels//2, [3, 3], stride=2, activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm, scope='conv1x1_before')

    with tf.variable_scope('second_branch'):

        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='conv1x1_after_pool')
        x = slim.conv2d(x, out_channels // 2, [3, 3], stride=1, activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm, scope='conv1x1_after')

    x = tf.concat([x, y], axis=3)
    return x


def block_plain(x, num_units,out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x = basic_unit_with_downsampling_plain(x,out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):

                x = basic_unit_plain(x)

    return x

def block(x, num_units,out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x = basic_unit_with_downsampling(x,out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):

                x = basic_unit(x)

    return x

def block_with_shuffle(x, num_units, L2_reg,training,out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x,L2_reg, training,out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)

                x = basic_unit(x,L2_reg,training)

        x = tf.concat([x, y], axis=3)

    return x
def block_plain_with_shuffle(x, num_units, L2_reg,training,out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling_plain(x,L2_reg, training,out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)

                x = basic_unit_plain(x,L2_reg,training)

        x = tf.concat([x, y], axis=3)

    return x



def preprocess( image):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.DATA.PIXEL_MEAN
        std = np.asarray(cfg.DATA.PIXEL_STD)

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean) * image_invstd

    return image
def simple_nn(inputs,L2_reg,training=True):
    inputs=preprocess(inputs)
    arg_scope = shufflenet_arg_scope(weight_decay=L2_reg)
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.batch_norm], is_training=training):
            with tf.variable_scope('ShuffleNetV2'):

                net = slim.conv2d(inputs, 24, [3, 3],stride=2, activation_fn=tf.nn.relu,
                                  normalizer_fn=slim.batch_norm, scope='init_conv')
                net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool1')
                # net = slim.separable_conv2d(net, 24, [3, 3], stride=2, activation_fn=tf.nn.relu,
                #                           normalizer_fn=slim.batch_norm, scope='init_conv_2', depth_multiplier=1)

                print('first conv shape', net.shape)
                net = block_plain(net, num_units=2, out_channels=64, scope='Stage2')
                print('2 conv shape', net.shape)
                net = block_plain(net, num_units=4, out_channels=128, scope='Stage3')
                print('3 conv shape', net.shape)
                net = block_plain(net, num_units=2, out_channels=256, scope='Stage4')
                print('4 conv shape', net.shape)

                ##use three layes feature in different scale 28x28 14x14 7x7

                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                net = slim.conv2d(net, cfg.MODEL.out_channel, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')

                net_out = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net_out = tf.identity(net_out, name='prediction')
    return net_out

