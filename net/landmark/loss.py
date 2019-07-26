#-*-coding:utf-8-*-

import tensorflow as tf
import math
from train_config import config as cfg
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
        #loss = tf.reduce_sum(losses, axis=[0,1])
        loss = tf.reduce_mean(losses, axis=[0,1])
        return loss


def _mse(landmarks, labels):
    
    return tf.reduce_mean(tf.square(landmarks-labels))/2.

def l1(landmarks, labels):

    return tf.reduce_mean(landmarks-labels)
def calculate_loss(predict,labels,scope):

    landmark_label = labels[:, 0:136]
    pose_label=labels[:, 136:139]
    #landmark_label = labels[:, 0:132]
    leye_cla_label = labels[:, 139]
    reye_cla_label = labels[:, 140]
    mouth_cla_label = labels[:, 141]
    landmark_predict = predict[:, 0:136]
    #landmark_predict = predict[:, 0:132]
    pose_predict=predict[:, 136:139]
    leye_cla_predict = predict[:, 139]
    reye_cla_predict = predict[:, 140]
    mouth_cla_predict = predict[:, 141]




    ###make crosssentropy
    loss_1 = _wing_loss(landmark_predict, landmark_label)

    loss_2 = _wing_loss(pose_predict, pose_label)

    leye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=leye_cla_predict,
                                                                       labels=leye_cla_label))
    reye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reye_cla_predict,
                                                                       labels=reye_cla_label))
    mouth_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mouth_cla_predict,
                                                                        labels=mouth_cla_label))

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

    loss=loss_1+loss_2

    regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')


    return loss, leye_loss, reye_loss, mouth_loss, leye_cla_accuracy, reye_cla_accuracy, mouth_cla_accuracy, regularization_losses

