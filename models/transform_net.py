import tensorflow as tf
from tensorflow import keras
# import os
# import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, '../utils'))
from utils import tf_utils


class Input_Transform_Net(keras.Model):
    def __init__(self, num_points, K=3):
        super(Input_Transform_Net).__init__(name='input_transform_net')
        self.K = K
        self.conv1 = tf_utils.Conv2D(64, [1, 3], padding='valid', strides=(1, 1), bn=True, name='conv1')
        self.conv2 = tf_utils.Conv2D(128, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv2')
        self.conv3 = tf_utils.Conv2D(256, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv3')
        self.maxpooling = keras.layers.MaxPool2D([num_points, 1], padding='valid')

        ## fully connected
        self.fc1 = tf_utils.FC(512, name='fc1')
        self.fc2 = tf_utils.FC(256, name='fc2')


    def call(self, inputs, training=None, mask=None):
        batch_size = inputs.get_shape()
        num_point = inputs.get_shape()
        inputs = tf.expand_dims(inputs, -1)
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.maxpooling(out)

        out = tf.reshape(out, [batch_size, -1])
        out = self.fc1(out)
        out = self.fc2(out)

        weights = tf.Variable(initial_value=tf.zeros([256, 3*self.K]), dtype=tf.float32)
        bias = tf.Variable(initial_value=tf.zeros([3*self.K]), dtype=tf.float32)
        bias += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)

        out = out@weights + bias
        transform = tf.reshape(out, [batch_size, 3, self.K])
        return transform


class Feature_Transform_Net(keras.Model):
    def __init__(self, num_points, K=3):
        super(Feature_Transform_Net).__init__(name='feature_transform_net')
        self.K = K
        self.conv1 = tf_utils.Conv2D(64, [1, 3], padding='valid', strides=(1, 1), bn=True, name='conv1')
        self.conv2 = tf_utils.Conv2D(128, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv2')
        self.conv3 = tf_utils.Conv2D(1024, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv3')
        self.maxpooling = keras.layers.MaxPool2D([num_points, 1], padding='valid')

        ## fully connected
        self.fc1 = tf_utils.FC(512, name='fc1')
        self.fc2 = tf_utils.FC(256, name='fc2')


    def call(self, inputs, training=None, mask=None):
        batch_size = inputs.get_shape()
        num_point = inputs.get_shape()
        inputs = tf.expand_dims(inputs, -1)
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.maxpooling(out)

        out = tf.reshape(out, [batch_size, -1])
        out = self.fc1(out)
        out = self.fc2(out)

        weights = tf.Variable(initial_value=tf.zeros([256, self.K*self.K]), dtype=tf.float32)
        bias = tf.Variable(initial_value=tf.zeros([self.K*self.K]), dtype=tf.float32)
        bias += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)

        out = out@weights + bias
        transform = tf.reshape(out, [batch_size, self.K, self.K])
        return transform


if __name__ == '__main__':
    print("test..")
