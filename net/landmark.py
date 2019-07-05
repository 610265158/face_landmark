# -*-coding:utf-8-*-

import tensorflow as tf
import math
import numpy as np
from train_config import config as cfg

from net.Resnet import resnet
from net.shufflenet import shufflenet_v2,shufflenet_v2_FPN
from net.Mobilenet import mobilenet


def preprocess( image):

    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.DATA.PIXEL_MEAN
        #std = np.asarray(cfg.DATA.PIXEL_STD)

        image_mean = tf.constant(mean, dtype=tf.float32)
        #image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean) #* image_invstd                   ###imagenet preprocess just centered the data

    return image


def simple_face(images,labels, L2_reg, training):
    images=preprocess(images)
    net_out = resnet(images, L2_reg, training)

    loss, leye_loss, reye_loss, mouth_loss, leye_cla_accuracy, \
    reye_cla_accuracy, mouth_cla_accuracy, l2_loss = calculate_loss(net_out, labels)

    return  loss, leye_loss, reye_loss, mouth_loss, leye_cla_accuracy, \
    reye_cla_accuracy, mouth_cla_accuracy, l2_loss









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
    return tf.reduce_mean(tf.square(landmarks - labels)) / 2.


def l1(landmarks, labels):
    return tf.reduce_mean(landmarks - labels)


def calculate_loss(predict, labels):
    landmark_label_reshaped = tf.reshape(labels[:, 0:136], [cfg.TRAIN.batch_size, 68, 2])
    pose_label = labels[:, 136:139]
    # landmark_label = labels[:, 0:132]
    leye_cla_label = labels[:, 139]
    reye_cla_label = labels[:, 140]
    mouth_cla_label = labels[:, 141]
    landmark_predict_reshaped = tf.reshape(predict[:, 0:136], [cfg.TRAIN.batch_size, 68, 2])
    # landmark_predict = predict[:, 0:132]
    pose_predict = predict[:, 136:139]
    leye_cla_predict = predict[:, 139]
    reye_cla_predict = predict[:, 140]
    mouth_cla_predict = predict[:, 141]

    face_boundary_label = tf.reshape(landmark_label_reshaped[:, 0:17, :], [cfg.TRAIN.batch_size, -1])
    l_eye_bow_label = tf.reshape(landmark_label_reshaped[:, 17:22, :], [cfg.TRAIN.batch_size, -1])
    r_eye_bow_label = tf.reshape(landmark_label_reshaped[:, 22:27, :], [cfg.TRAIN.batch_size, -1])
    nose_label = tf.reshape(landmark_label_reshaped[:, 27:36, :], [cfg.TRAIN.batch_size, -1])
    l_eye_label = tf.reshape(landmark_label_reshaped[:, 36:42, :], [cfg.TRAIN.batch_size, -1])
    r_eye_label = tf.reshape(landmark_label_reshaped[:, 42:48, :], [cfg.TRAIN.batch_size, -1])
    mouth_label = tf.reshape(landmark_label_reshaped[:, 48:, :], [cfg.TRAIN.batch_size, -1])

    face_boundary_predict = tf.reshape(landmark_predict_reshaped[:, 0:17, :], [cfg.TRAIN.batch_size, -1])
    l_eye_bow_predict = tf.reshape(landmark_predict_reshaped[:, 17:22, :], [cfg.TRAIN.batch_size, -1])
    r_eye_bow_predict = tf.reshape(landmark_predict_reshaped[:, 22:27, :], [cfg.TRAIN.batch_size, -1])
    nose_predict = tf.reshape(landmark_predict_reshaped[:, 27:36, :], [cfg.TRAIN.batch_size, -1])
    l_eye_predict = tf.reshape(landmark_predict_reshaped[:, 36:42, :], [cfg.TRAIN.batch_size, -1])
    r_eye_predict = tf.reshape(landmark_predict_reshaped[:, 42:48, :], [cfg.TRAIN.batch_size, -1])
    mouth_predict = tf.reshape(landmark_predict_reshaped[:, 48:, :], [cfg.TRAIN.batch_size, -1])

    ###make crosssentropy
    loss_1 = _wing_loss(face_boundary_predict, face_boundary_label)
    loss_2 = _wing_loss(l_eye_bow_predict, l_eye_bow_label)
    loss_3 = _wing_loss(r_eye_bow_predict, r_eye_bow_label)
    loss_4 = _wing_loss(nose_predict, nose_label)
    loss_5 = _wing_loss(l_eye_predict, l_eye_label)
    loss_6 = _wing_loss(r_eye_predict, r_eye_label)
    loss_7 = _wing_loss(mouth_predict, mouth_label)

    loss_8 = _mse(pose_predict, pose_label)
    # loss_1=_wing_loss(landmark_predict,landmark_label)/cfg.TRAIN.batch_size
    # tf.add_to_collection('%smutiloss'%scope,tf.summary.scalar('face_boundary_loss', loss_1))
    # tf.add_to_collection('%smutiloss'%scope, tf.summary.scalar('l_eye_bow_loss', loss_2))
    # tf.add_to_collection('%smutiloss'%scope, tf.summary.scalar('r_eye_bow_loss', loss_3))
    # tf.add_to_collection('%smutiloss'%scope, tf.summary.scalar('nose_loss', loss_4))
    # tf.add_to_collection('%smutiloss'%scope, tf.summary.scalar('l_eye_loss', loss_5))
    # tf.add_to_collection('%smutiloss'%scope, tf.summary.scalar('r_eye_loss', loss_6))
    # tf.add_to_collection('%smutiloss'%scope, tf.summary.scalar('mouth_loss', loss_7))

    leye_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=leye_cla_predict,
                                                                      labels=leye_cla_label)) / cfg.TRAIN.batch_size / 2.
    reye_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=reye_cla_predict,
                                                                      labels=reye_cla_label)) / cfg.TRAIN.batch_size / 2.
    mouth_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=mouth_cla_predict,
                                                                       labels=mouth_cla_label)) / cfg.TRAIN.batch_size / 2.

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

    loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7 + loss_8

    regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')

    return loss, leye_loss, reye_loss, mouth_loss, leye_cla_accuracy, reye_cla_accuracy, mouth_cla_accuracy, regularization_losses

