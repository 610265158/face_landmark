#-*-coding:utf-8-*-
from net.inception_resnet_v2.inception_resnet_v2 import inception_resnet_v2,inception_resnet_v2_arg_scope
import tensorflow.contrib.slim as slim
import tensorflow as tf

from train_config import config as cfg

def inception_resnet(input, L2_reg, training,trainable=True):

    arg_scope = inception_resnet_v2_arg_scope(weight_decay=L2_reg)
    with slim.arg_scope(arg_scope):
        net, end_points = inception_resnet_v2(input, is_training=training, num_classes=None)

        # Global average pooling.
        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)


        net = slim.conv2d(net, cfg.MODEL.out_channel, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')

        net_out = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    act_net = tf.nn.sigmoid(net_out)
    act_net=tf.identity(act_net,name='prediction')
    return net_out,act_net
