#-*-coding:utf-8-*-
from net.resnet.resnet_v2 import resnet_v2_50,resnet_arg_scope
import tensorflow.contrib.slim as slim
import tensorflow as tf

from train_config import config as cfg

def resnet(input, L2_reg, training,trainable=True):

    arg_scope = resnet_arg_scope(weight_decay=L2_reg)
    with slim.arg_scope(arg_scope):
        net, end_points = resnet_v2_50(input, is_training=training, global_pool=False, num_classes=None)

        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        net = slim.conv2d(net, cfg.MODEL.out_channel, [1, 1], activation_fn=None,trainable=trainable,
                          normalizer_fn=None, scope='logits')

        net_out = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net_out = tf.identity(net_out, name='prediction')
    return net_out

