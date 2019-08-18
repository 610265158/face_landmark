# -*-coding:utf-8-*-

import tensorflow as tf
import tensorflow.contrib.slim as slim

import math
from train_config import config as cfg


from lib.core.model.resnet.resnet_v1 import resnet_arg_scope,resnet_v1_50

def preprocess( image):

    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.DATA.PIXEL_MEAN

        image_mean = tf.constant(mean, dtype=tf.float32)

        image = image - image_mean###imagenet preprocess just centered the data

    return image


def simple_face(images,labels, L2_reg, training):
    images=preprocess(images)
    arg_scope = resnet_arg_scope(weight_decay=L2_reg)
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.batch_norm], is_training=training):
            net, end_points = resnet_v1_50(images, is_training=training, global_pool=False, num_classes=None)

            for k, v in end_points.items():
                print(k, v)


            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            net = slim.conv2d(net, cfg.MODEL.out_channel, [1, 1], activation_fn=None,
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
        # loss = tf.reduce_sum(losses, axis=[0,1])
        loss = tf.reduce_mean(losses, axis=[0, 1])
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

    landmark_predict = predict[:, 0:136]
    pose_predict = predict[:, 136:139]
    leye_cla_predict = predict[:, 139]
    reye_cla_predict = predict[:, 140]
    mouth_cla_predict = predict[:, 141]


    loss = _wing_loss(landmark_predict, landmark_label)

    loss_pose = _mse(pose_predict, pose_label)

    loss=loss_pose+loss

    leye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=leye_cla_predict,
                                                                      labels=leye_cla_label))
    reye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reye_cla_predict,
                                                                      labels=reye_cla_label))
    mouth_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mouth_cla_predict,
                                                                       labels=mouth_cla_label))
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

    return loss, leye_loss, reye_loss, mouth_loss, leye_cla_accuracy, reye_cla_accuracy, mouth_cla_accuracy, regularization_losses

