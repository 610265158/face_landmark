import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg

from lib.core.model.mobilenet import mobilenet_v2_035,training_scope
from net.resnet.basemodel import resnet_arg_scope

def mobilenet_ssd(image,L2_reg,is_training=True,data_format='NHWC'):


    assert 'MobilenetV2' in cfg.MODEL.net_structure
    if cfg.TRAIN.lock_basenet_bn:
        arg_scope = training_scope(weight_decay=L2_reg, is_training=False)
    else:
        arg_scope = training_scope(weight_decay=L2_reg, is_training=is_training)


    with tf.contrib.slim.arg_scope(arg_scope):
        _,endpoint = mobilenet_v2_035(image,is_training=is_training,num_classes=None,base_only=True)

    for k,v in endpoint.items():
        print('mobile backbone output:',k,v)

    mobilebet_fms=[endpoint['layer_4'],endpoint['layer_7'],endpoint['layer_14'],endpoint['layer_19']]#layer_18?

    print('mobile backbone output:',mobilebet_fms)
    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training, bn_trainable=True,
                                         data_format=data_format)):

        # model = block(resnet_fms[-1], num_units=2, out_channels=512, scope='extra_Stage1')
        # resnet_fms.append(model)
        # model = block(model, num_units=2, out_channels=512, scope='extra_Stage2')
        # resnet_fms.append(model)
        net = slim.conv2d(mobilebet_fms[-1], 128, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_1_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_1_2')
        mobilebet_fms.append(net)
        net = slim.conv2d(net, 128, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_2_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_2_2')
        mobilebet_fms.append(net)
        print('extra backbone output:', mobilebet_fms)

    return mobilebet_fms
