import tensorflow as tf
import os
import sys
from tensorflow import keras
import numpy as np
from models.transform_net import Input_Transform_Net, Feature_Transform_Net
from utils import tf_utils

class ClsModel(keras.Model):

    def __init__(self, num_points):
        super(ClsModel).__init__(name='clsModel')
        self.input_transform = Input_Transform_Net(num_points)
        self.feature_transform = Feature_Transform_Net(num_points)
        self.conv1 = tf_utils.Conv2D(64, [1, 3], padding='valid', strides=(1, 1), bn=True, name='conv1')
        self.conv2 = tf_utils.Conv2D(64, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv1')

        self.conv3 = tf_utils.Conv2D(64, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv1')
        self.conv4 = tf_utils.Conv2D(128, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv1')
        self.conv5 = tf_utils.Conv2D(1024, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv1')

        self.maxpooling = keras.layers.MaxPool2D([num_points, 1], padding='valid')

        self.fc1 = tf_utils.FC(512, name='fc1')
        self.drop1 = keras.layers.Dropout(0.7)
        self.fc2 = tf_utils.FC(256, name='fc2')
        self.drop2 = keras.layers.Dropout(0.7)
        self.fc3 = tf_utils.FC(40, name='fc3', activation=False)
        self.drop4 = keras.layers.Dropout(0.7)

    def call(self, inputs):
        batch_size = inputs.shape[0]
        num_points = inputs.shape[1]
        # input_transform
        end_points = {}
        transform = self.input_transform(inputs)
        after_transform = tf.matmul(inputs, transform)
        input_image = tf.expand_dims(after_transform, -1)
        out = self.conv1(input_image)
        out = self.conv2(input_image)
        # feature_transform
        transform2 = self.feature_transform(out)
        end_points['transform'] = transform
        out_transform = tf.matmul(tf.squeeze(out, axis=2), transform2)
        out_transform = tf.expand_dims(out_transform, [2])

        out_transform = self.maxpooling(out_transform)
        out_transform = tf.reshape(out_transform, [batch_size, -1])
        out_transform = self.fc1(out_transform)
        out_transform = self.drop1(out_transform)
        out_transform = self.fc2(out_transform)
        out_transform = self.drop2(out_transform)
        out_transform = self.fc3(out_transform)
        out_transform = self.drop3(out_transform)
        return out_transform, end_points

def get_loss(pred, label, end_points, reg_weights=0.001):
    loss = tf.losses.sparse_categorical_crossentropy(y_pred=pred, y_true=label)
    cls_loss = tf.reduce_mean(loss)

    tf.summary.scalar('cls_loss', cls_loss)
    ## transform
    transform = end_points['transform']
    K = transform.shape[1]
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat_loss', mat_diff_loss)
    return cls_loss + mat_diff_loss * reg_weights
