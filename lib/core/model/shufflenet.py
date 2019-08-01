#-*-coding:utf-8-*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from train_config import config as cfg

from lib.core.model.resnet.resnet_v2 import resnet_arg_scope




def block(x, num_units, L2_reg,training,out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x,L2_reg, training,out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)

                x = basic_unit(x,L2_reg,training)

        x = tf.concat([x, y], axis=3)

    return x

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

def basic_unit(x,L2_reg,training):
    in_channels = x.shape[3].value

    x = slim.conv2d(x, in_channels, [1, 1], stride=1, activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm, scope='conv1x1_before')

    x = slim.separable_conv2d(x, in_channels, [3, 3], stride=1, activation_fn=None,
                              normalizer_fn=slim.batch_norm, scope='depthwise', depth_multiplier=1)

    x = slim.conv2d(x, in_channels, [1, 1], stride=1, activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm, scope='conv1x1_after')
    return x

def basic_unit_with_downsampling(x, L2_reg,training,out_channels=None):
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

    return x, y


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


def shufflenet_v2(inputs,L2_reg,training=True):
    arg_scope = shufflenet_arg_scope(weight_decay=L2_reg)
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.batch_norm], is_training=training):
            with tf.variable_scope('ShuffleNetV2'):

                net = slim.conv2d(inputs, 24, [3, 3],stride=2, activation_fn=tf.nn.relu,
                                  normalizer_fn=slim.batch_norm, scope='init_conv')

                #model = tf.nn.max_pool(model, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool1')
                net = slim.separable_conv2d(net, 24, [3, 3], stride=2, activation_fn=tf.nn.relu,
                                          normalizer_fn=slim.batch_norm, scope='init_conv_2', depth_multiplier=1)
                print('first conv shape', net.shape)
                x = block(net, num_units=4, L2_reg=L2_reg, training=training, out_channels=48, scope='Stage2')
                print('Stage2 shape', x.shape)
                x = block(x, num_units=8, L2_reg=L2_reg, training=training,out_channels=96, scope='Stage3')
                print('Stage3 shape', x.shape)
                x = block(x, num_units=4, L2_reg=L2_reg, training=training,out_channels=192, scope='Stage4')
                print('Stage4 shape', x.shape)

                x = slim.conv2d(x,512, [1, 1], activation_fn=tf.nn.relu,
                                  normalizer_fn=slim.batch_norm, scope='last_conv')

                # Global average pooling.
                net = tf.reduce_mean(x, [1, 2], name='pool5', keep_dims=True)
                net = slim.conv2d(net, cfg.MODEL.out_channel, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')

                net_out = tf.squeeze(net, [1, 2], name='prediction')

    return net_out








def shufflenet_v2_FPN(inputs,L2_reg,training=True):
    arg_scope = shufflenet_arg_scope(weight_decay=L2_reg)
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.batch_norm], is_training=training):
            with tf.variable_scope('ShuffleNetV2'):

                net = slim.conv2d(inputs, 24, [3, 3],stride=2, activation_fn=tf.nn.relu,
                                  normalizer_fn=slim.batch_norm, scope='init_conv')
                #model = tf.nn.max_pool(model, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool1')
                net = slim.separable_conv2d(net, 24, [3, 3], stride=2, activation_fn=tf.nn.relu,
                                          normalizer_fn=slim.batch_norm, scope='init_conv_2', depth_multiplier=1)
                print('init conv shape', net.shape)
                x1 = block(net, num_units=4, L2_reg=L2_reg, training=training, out_channels=48, scope='Stage2')
                print('Stage2 shape', x1.shape)
                x2 = block(x1, num_units=8, L2_reg=L2_reg, training=training, out_channels=96,scope='Stage3')
                print('Stage3 shape', x2.shape)
                x3 = block(x2, num_units=4, L2_reg=L2_reg, training=training,out_channels=192, scope='Stage4')
                print('Stage4 shape', x3.shape)


                ##use three layes feature in different scale 28x28 14x14 7x7

                net= tf.image.resize_images(net,(28,28))
                x2=tf.image.resize_images(x2,(28,28))
                x3 = tf.image.resize_images(x3, (28, 28))
                fpn_feature=tf.concat([net,x1,x2,x3],axis=3)


                x = slim.conv2d(fpn_feature, 256, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm, scope='last_conv_1x1_before')
                x = slim.separable_conv2d(x, 256, [3, 3], stride=2, activation_fn=None,
                                    normalizer_fn=None, scope='last_conv_1', depth_multiplier=1)
                x = slim.separable_conv2d(x, 256, [3, 3], stride=2, activation_fn=tf.nn.relu,
                                          normalizer_fn=slim.batch_norm, scope='last_conv_2', depth_multiplier=1)
                x = slim.conv2d(x, 256, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm, scope='last_conv_1x1_after')

                # Global average pooling.
                net = tf.reduce_mean(x, [1, 2], name='pool5', keep_dims=True)
                net = slim.conv2d(net, cfg.MODEL.out_channel, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')

                net_out = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net_out = tf.identity(net_out, name='prediction')
    return net_out

