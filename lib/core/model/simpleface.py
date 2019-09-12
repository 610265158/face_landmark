# -*-coding:utf-8-*-

import tensorflow as tf
import tensorflow.contrib.slim as slim

import math
import numpy as np

from train_config import config as cfg


from lib.core.model.resnet.resnet_v1 import resnet_arg_scope,resnet_v1_50
from lib.core.model.mobilenet.mobilenet_v2 import mobilenet_v2_050
from lib.core.model.shufflenet import shufflenet_v2

def preprocess(image):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.DATA.PIXEL_MEAN
        std = np.asarray(cfg.DATA.PIXEL_STD)

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean)*image_invstd

    return image


def simple_face(images,labels, training):
    images=preprocess(images)
    if 'ShuffleNetV2'in cfg.MODEL.net_structure:
        net, end_points=shufflenet_v2(images,is_training=training,depth_multiplier='1.0')
        for k, v in end_points.items():
            print(k, v)
        print(net)
        s1 = tf.reduce_mean(end_points['stage2'], [1, 2], name='pool1', keep_dims=True)
        s2 = tf.reduce_mean(end_points['stage3'], [1, 2], name='pool2', keep_dims=True)
        s3 = tf.reduce_mean(end_points['stage4'], [1, 2], name='pool3', keep_dims=True)
        multi_scale = tf.concat([s1, s2, s3], 3)

    else:
        arg_scope = resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            with slim.arg_scope([slim.batch_norm], is_training=training):

                if 'resnet' in cfg.MODEL.net_structure:
                    net, end_points = resnet_v1_50(images, is_training=training, global_pool=False, num_classes=None)
                ###muti scale there for resnet
                    for k, v in end_points.items():
                        print(k, v)
                    multi_scale=net

                elif 'MobilenetV2' in cfg.MODEL.net_structure:
                    net, end_points =mobilenet_v2_050(images,base_only=True,is_training=training,finegrain_classification_mode=False)
                    for k, v in end_points.items():
                        print(k, v)

                        s1 = tf.reduce_mean(end_points['layer_7/output'], [1, 2], name='pool1', keep_dims=True)
                        s2 = tf.reduce_mean(end_points['layer_14/output'], [1, 2], name='pool2', keep_dims=True)
                        s3 = tf.reduce_mean(end_points['layer_18/output'], [1, 2], name='pool3', keep_dims=True)
                        multi_scale = tf.concat([s1, s2, s3], 3)


    net = slim.conv2d(multi_scale, cfg.MODEL.out_channel, [1, 1], activation_fn=None,
                      normalizer_fn=None, scope='logits')
    net_out = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net_out = tf.identity(net_out, name='prediction')

    regression_loss, leye_loss, reye_loss, mouth_loss, leye_cla_accuracy, \
    reye_cla_accuracy, mouth_cla_accuracy, l2_loss = calculate_loss(net_out, labels)

    return regression_loss, leye_loss, reye_loss, mouth_loss, leye_cla_accuracy, \
    reye_cla_accuracy, mouth_cla_accuracy, l2_loss

def _heatmap_branch(fm,repeat=2,scope='heatmaps'):

    with tf.variable_scope(scope):
        for i in range(repeat):
            fm = slim.conv2d_transpose(fm, 256, [3, 3],stride=2,  scope='deconv%d'%i)
        net = slim.conv2d(fm, 68, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='%soutput'%scope)

    return net


def _wing_loss(landmarks, labels, w=10.0, epsilon=2.0):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, landmarks].  landmarks means x1,x2,x3,x4...y1,y2,y3,y4   1-D
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    with tf.name_scope('wing_loss'):
        x = landmarks - labels
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.log(1.0 + absolute_x / epsilon),
            absolute_x - c
        )
        losses=losses*cfg.DATA.weights
        loss = tf.reduce_sum(tf.reduce_mean(losses, axis=[0]))
        return loss


def _mse(landmarks, labels):

    return tf.reduce_mean(0.5*tf.square(landmarks - labels))


def l1(landmarks, labels):
    return tf.reduce_mean(landmarks - labels)


def calculate_loss(predict, labels):
    landmark_label = labels[:, 0:136]
    pose_label = labels[:, 136:139]
    leye_cla_label = labels[:, 139]
    reye_cla_label = labels[:, 140]
    mouth_cla_label = labels[:, 141]
    big_mouth_cla_label = labels[:, 142]

    landmark_predict = predict[:, 0:136]
    pose_predict = predict[:, 136:139]
    leye_cla_predict = predict[:, 139]
    reye_cla_predict = predict[:, 140]
    mouth_cla_predict = predict[:, 141]
    big_mouth_cla_predict = predict[:, 142]

    loss = _wing_loss(landmark_predict, landmark_label)

    loss_pose = _mse(pose_predict, pose_label)

    loss=loss_pose+loss

    leye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=leye_cla_predict,
                                                                      labels=leye_cla_label))
    reye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reye_cla_predict,
                                                                      labels=reye_cla_label))
    mouth_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mouth_cla_predict,
                                                                       labels=mouth_cla_label))
    mouth_loss_big = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=big_mouth_cla_predict,
                                                                        labels=big_mouth_cla_label))
    mouth_loss=mouth_loss+mouth_loss_big
    ###make crosssentropy
    leye_cla_correct_prediction = tf.equal(
        tf.cast(tf.greater_equal(tf.nn.sigmoid(leye_cla_predict), 0.5), tf.int32),
        tf.cast(leye_cla_label, tf.int32))
    leye_cla_accuracy = tf.reduce_mean(tf.cast(leye_cla_correct_prediction, tf.float32))

    reye_cla_correct_prediction = tf.equal(
        tf.cast(tf.greater_equal(tf.nn.sigmoid(reye_cla_predict), 0.5), tf.int32),
        tf.cast(reye_cla_label, tf.int32))
    reye_cla_accuracy = tf.reduce_mean(tf.cast(reye_cla_correct_prediction, tf.float32))
    mouth_cla_correct_prediction = tf.equal(
        tf.cast(tf.greater_equal(tf.nn.sigmoid(mouth_cla_predict), 0.5), tf.int32),
        tf.cast(mouth_cla_label, tf.int32))
    mouth_cla_accuracy = tf.reduce_mean(tf.cast(mouth_cla_correct_prediction, tf.float32))


    regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')


    if cfg.MODEL.pruning:
        bn_l1_loss = []
        bn_reg = slim.l1_regularizer(cfg.MODEL.pruning_bn_reg)
        variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=cfg.MODEL.net_structure)
        for var in variables_restore:
            if 'beta' in var.name:
                bn_l1_loss.append(bn_reg(var))
        l1_loss = tf.add_n(bn_l1_loss, name='l1_loss')

        regularization_losses = regularization_losses + l1_loss


    return loss, leye_loss, reye_loss, mouth_loss, leye_cla_accuracy, reye_cla_accuracy, mouth_cla_accuracy, regularization_losses

