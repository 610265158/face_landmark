#-*-coding:utf-8-*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from train_config import config as cfg

from net.mobilenet.mobilenet_v2 import mobilenet_v2_035,training_scope





def mobilenet(input, L2_reg, training,trainable=True):

    arg_scope = training_scope(weight_decay=L2_reg, is_training=training)

    with tf.contrib.slim.arg_scope(arg_scope):
        _,endpoint = mobilenet_v2_035(input, is_training=training, num_classes=None, base_only=True)

        # for k, v in endpoint.items():
        #     print('mobile backbone output:', k, v)

        net=endpoint['layer_19']
        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        net = slim.conv2d(net, cfg.MODEL.out_channel, [1, 1], activation_fn=None, trainable=trainable,
                          normalizer_fn=None, scope='logits')

        net_out = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net_out = tf.identity(net_out, name='prediction')
    return net_out

