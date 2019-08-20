import tensorflow as tf
from tensorflow import keras
# import os
# import sys
import numpy as np
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, '../utils'))
from utils import tf_utils


class Input_Transform_Net(keras.Model):
    def __init__(self, num_points, K=3):
        super(Input_Transform_Net, self).__init__(name='input_transform_net')
        self.K = K
        self.num_points = num_points

        self.conv1 = tf_utils.MyConv(64, [1, 3], padding='valid', strides=(1, 1), bn=True, name='iconv1')
        self.conv2 = tf_utils.MyConv(128, [1, 1], padding='valid', strides=(1, 1), bn=True, name='iconv2')
        self.conv3 = tf_utils.MyConv(256, [1, 1], padding='valid', strides=(1, 1), bn=True, name='iconv3')
        self.maxpooling = keras.layers.MaxPool2D([self.num_points, 1], padding='valid')
        self.flatten = keras.layers.Flatten()
        ## fully connected
        self.fc1 = tf_utils.FC(512, name='ifc1')
        self.fc2 = tf_utils.FC(256, name='ifc2')

        self.w = tf.Variable(initial_value=tf.zeros([256, 3 * self.K]), dtype=tf.float32)
        self.b = tf.Variable(initial_value=tf.zeros([3 * self.K]), dtype=tf.float32)
        self.b.assign_add(tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32))


    def call(self, inputs, training=None):

        inputs = tf.expand_dims(inputs, -1)
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpooling(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)

        out = tf.matmul(out, self.w) + self.b
        transform = tf.reshape(out, [-1, 3, self.K])
        return transform


class Feature_Transform_Net(keras.Model):

    def __init__(self, num_points, K=3):
        super(Feature_Transform_Net, self).__init__(name='feature_transform_net')
        self.K = K
        self.num_points = num_points
        self.conv1 = tf_utils.MyConv(64, [1, 1], padding='valid', strides=(1, 1), bn=True, name='fconv1')
        self.conv2 = tf_utils.MyConv(128, [1, 1], padding='valid', strides=(1, 1), bn=True, name='fconv2')
        self.conv3 = tf_utils.MyConv(1024, [1, 1], padding='valid', strides=(1, 1), bn=True, name='fconv3')
        self.maxpooling = keras.layers.MaxPool2D([self.num_points, 1], padding='valid')
        self.flatten = keras.layers.Flatten()
        ## fully connected
        self.fc1 = tf_utils.FC(512, name='ffc1')
        self.fc2 = tf_utils.FC(256, name='ffc2')

        self.w = tf.Variable(initial_value=tf.zeros([256, self.K * self.K]), dtype=tf.float32)
        self.b = tf.Variable(initial_value=tf.zeros([self.K * self.K]), dtype=tf.float32)
        self.b.assign_add(tf.constant(np.eye(K).flatten(), dtype=tf.float32))


    def call(self, inputs, training=None):


            out = self.conv1(inputs)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.maxpooling(out)

            out = self.flatten(out)
            out = self.fc1(out)
            out = self.fc2(out)

            out = out@self.w + self.b
            transform = tf.reshape(out, [-1, self.K, self.K])
            return transform


if __name__ == '__main__':
    print("test..")
    model = Feature_Transform_Net(1024, 64)
    model.build(input_shape=(None, 1024, 1, 64))
