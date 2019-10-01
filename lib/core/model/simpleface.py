# -*-coding:utf-8-*-
import sys
sys.path.append('.')
import tensorflow as tf

import math
import numpy as np

from train_config import config as cfg




from lib.core.model.shufflenet_plus import ShuffleNetPlus




class SimpleFaceHead(tf.keras.Model):
    def __init__(self,):
        super(SimpleFaceHead, self).__init__()


        self.output_size=cfg.MODEL.out_channel


        self.conv=tf.keras.layers.Dense(self.output_size,
                                            use_bias=True)


    def call(self, inputs, training=False):


        output=self.conv(inputs)

        return output





class SimpleFace(tf.keras.Model):

    def __init__(self):
        super(SimpleFace, self).__init__()


        self.backbone=ShuffleNetPlus()

        self.head=SimpleFaceHead()



        self.pool1=tf.keras.layers.GlobalAveragePooling2D()
        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.pool3 = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=False):

        inputs=self.preprocess(inputs)


        net,end_points=self.backbone(inputs,training=training)


        s1 = self.pool1(end_points['layer7'])
        s2 = self.pool2(end_points['layer15'])
        s3 = self.pool3(end_points['layer20'])

        multi_scale = tf.concat([s1, s2, s3], 1)

        out_put=self.head(multi_scale,training=training)

        return out_put

    def preprocess(self,image):

        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.DATA.PIXEL_MEAN
        std = np.asarray(cfg.DATA.PIXEL_STD)

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean) * image_invstd

        return image

    def load_weights(self):
        pass
    def inference(self):
        pass









if __name__=='__main__':

    import numpy as np

    model = SimpleFace()

    image=np.zeros(shape=(1,160,160,3),dtype=np.float32)
    x=model(image)

    model.summary()






def simple_face(images,labels, training):





    if 'ShuffleNetV2PLUS' in cfg.MODEL.net_structure:

        model=ShuffleNetPlus()

        net, end_points = model(images, training=training)

        #net, end_points=shufflenet_v2(images,is_training=training,depth_multiplier='1.0')
        for k, v in end_points.items():
            print(k, v)
        print(net)
        s1 = tf.reduce_mean(end_points['layer7'], [1, 2], name='pool1', keep_dims=True)
        s2 = tf.reduce_mean(end_points['layer15'], [1, 2], name='pool2', keep_dims=True)
        s3 = tf.reduce_mean(end_points['layer20'], [1, 2], name='pool3', keep_dims=True)
        multi_scale = tf.concat([s1, s2, s3], 3)

    elif 'ShuffleNetV2' in cfg.MODEL.net_structure:

        model=ShuffleNetPlus()

        net, end_points = model(images, training=training)

        #net, end_points=shufflenet_v2(images,is_training=training,depth_multiplier='1.0')
        for k, v in end_points.items():
            print(k, v)
        print(net)
        s1 = tf.reduce_mean(end_points['layer7'], [1, 2], name='pool1', keep_dims=True)
        s2 = tf.reduce_mean(end_points['layer15'], [1, 2], name='pool2', keep_dims=True)
        s3 = tf.reduce_mean(end_points['layer20'], [1, 2], name='pool3', keep_dims=True)
        multi_scale = tf.concat([s1, s2, s3], 3)

    else:
        arg_scope = resnet_arg_scope()
        with sl

    net = slim.conv2d(multi_scale, cfg.MODEL.out_channel-2, [1, 1], activation_fn=None,
                      normalizer_fn=None, scope='logits')
    net_out = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net_out = tf.identity(net_out, name='prediction')

    loss, leye_loss, reye_loss, mouth_loss, eyeglasses_loss, gender_loss, mouth_slightly_loss,smile_loss, l2_loss = calculate_loss(net_out, labels)

    return loss, leye_loss, reye_loss, mouth_loss, eyeglasses_loss, gender_loss, mouth_slightly_loss,smile_loss, l2_loss


def _wing_loss(landmarks, labels, w=10.0, epsilon=2.0,weights=1.):
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
        loss = tf.reduce_sum(tf.reduce_mean(losses*weights, axis=[0]))
        return loss


def _mse(landmarks, labels,weights=1.):

    return tf.reduce_mean(0.5*tf.square(landmarks - labels)*weights)


def l1(landmarks, labels):
    return tf.reduce_mean(landmarks - labels)


def calculate_loss(predict, labels):
    
    
    
    weights_keypoint = tf.cast(labels[:, 147],tf.bool)
    weights_attr = tf.cast(labels[:, 148],tf.bool)

    label_keypoints=tf.boolean_mask(labels,weights_keypoint)
    predict_keypoints = tf.boolean_mask(predict, weights_keypoint)

    label_attr=tf.boolean_mask(labels,weights_attr)
    predict_attr = tf.boolean_mask(predict, weights_attr)    
    
    
    landmark_label =      label_keypoints[:, 0:136]
    pose_label =          label_keypoints[:, 136:139]
    leye_cls_label =      label_keypoints[:, 139]
    reye_cls_label =      label_keypoints[:, 140]
    mouth_cls_label =     label_keypoints[:, 141]
    big_mouth_cls_label = label_keypoints[:, 142]

    eyeglasses_cls_label = label_attr[:, 143]
    gender_cls_label =     label_attr[:, 144]
    mouth_slightly_open_cls_label = label_attr[:, 145]
    smile_cls_label =      label_attr[:, 146]

    

    landmark_predict =     predict_keypoints[:, 0:136]
    pose_predict =         predict_keypoints[:, 136:139]
    leye_cls_predict =     predict_keypoints[:, 139]
    reye_cls_predict =     predict_keypoints[:, 140]
    mouth_cls_predict =     predict_keypoints[:, 141]
    big_mouth_cls_predict = predict_keypoints[:, 142]

    eyeglasses_cls_predict = predict_attr[:, 143]
    gender_cls_predict =     predict_attr[:, 144]
    mouth_slightly_open_cls_predict = predict_attr[:, 145]
    smile_cls_predict = predict_attr[:, 146]









    loss = _wing_loss(landmark_predict, landmark_label)

    loss_pose = _mse(pose_predict, pose_label)

    loss=(loss_pose+loss)

    leye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=leye_cls_predict,
                                                                      labels=leye_cls_label) )
    reye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reye_cls_predict,
                                                                      labels=reye_cls_label))
    mouth_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mouth_cls_predict,
                                                                       labels=mouth_cls_label))
    mouth_loss_big = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=big_mouth_cls_predict,
                                                                        labels=big_mouth_cls_label))
    mouth_loss=mouth_loss+mouth_loss_big



    norm_v=tf.maximum(tf.reduce_sum(tf.cast(weights_attr,tf.float32)),1.)

    eyeglasses_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=eyeglasses_cls_predict,
                                                                      labels=eyeglasses_cls_label) )/norm_v
    gender_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=gender_cls_predict,
                                                                       labels=gender_cls_label))/norm_v
    mouth_slightly_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=mouth_slightly_open_cls_predict,
                                                                        labels=mouth_slightly_open_cls_label) )/norm_v
    smile_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=smile_cls_predict,
                                                                           labels=smile_cls_label))/norm_v

    # ##make crosssentropy
    # leye_cla_correct_prediction = tf.equal(
    #     tf.cast(tf.greater_equal(tf.nn.sigmoid(leye_cls_predict), 0.5), tf.int32),
    #     tf.cast(leye_cla_label, tf.int32))
    # leye_cla_accuracy = tf.reduce_mean(tf.cast(leye_cla_correct_prediction, tf.float32))
    #
    # reye_cla_correct_prediction = tf.equal(
    #     tf.cast(tf.greater_equal(tf.nn.sigmoid(reye_cla_predict), 0.5), tf.int32),
    #     tf.cast(reye_cla_label, tf.int32))
    # reye_cla_accuracy = tf.reduce_mean(tf.cast(reye_cla_correct_prediction, tf.float32))
    # mouth_cla_correct_prediction = tf.equal(
    #     tf.cast(tf.greater_equal(tf.nn.sigmoid(mouth_cla_predict), 0.5), tf.int32),
    #     tf.cast(mouth_cla_label, tf.int32))
    # mouth_cla_accuracy = tf.reduce_mean(tf.cast(mouth_cla_correct_prediction, tf.float32))



    #### l2 regularization_losses
    l2_loss = []
    l2_reg = tf.keras.regularizers.l2(cfg.TRAIN.weight_decay_factor)
    variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    for var in variables_restore:
        if 'weight' in var.name:
            l2_loss.append(l2_reg(var))
    regularization_losses = tf.add_n(l2_loss, name='l1_loss')



    if cfg.MODEL.pruning:
        bn_l1_loss = []
        bn_reg = slim.l1_regularizer(cfg.MODEL.pruning_bn_reg)
        variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=cfg.MODEL.net_structure)
        for var in variables_restore:
            if 'beta' in var.name:
                bn_l1_loss.append(bn_reg(var))
        l1_loss = tf.add_n(bn_l1_loss, name='l1_loss')

        regularization_losses = regularization_losses + l1_loss


    return loss, leye_loss, reye_loss, mouth_loss, eyeglasses_loss, gender_loss, mouth_slightly_loss,smile_loss, regularization_losses


