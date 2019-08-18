import tensorflow as tf
from tensorflow import keras
import math
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops

class Conv2D(keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernal_size,
                 padding,
                 strides,
                 name,
                 use_xavier=True,
                 stddev=1e-3,
                 weight_decay=0.0,
                 bn=False,
                 ):
        super(Conv2D).__init__(name=name)
        self.check = bn
        if use_xavier:
            self.conv2d = keras.layers.Conv2D(filters,
                                              kernal_size,
                                              padding=padding,
                                              strides=strides,
                                              kernel_regularizer=keras.regularizers.l2(weight_decay)
                                             )
        else:
            self.conv2d = keras.layers.Conv2D(filters,
                                              kernal_size,
                                              padding=padding,
                                              strides=strides,
                                              kernel_initializer=keras.initializers.TruncatedNormal(stddev=stddev),
                                              kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn = keras.layers.BatchNormalization()
        self.activation = keras.layers.Activation('relu')

    def call(self, inputs):

        output = self.conv2d(inputs)
        if self.check:
            output = self.bn(inputs)
        output = self.activation(output)
        return output

class FC(keras.layers.Layer):

    def __init__(self,
                 outdim,
                 name,
                 activation=True,
                 bn=True):
        super(FC).__init__(name=name)
        self.check = bn
        self.activation = activation
        self.fc = keras.layers.Dense(outdim)
        self.bn = keras.layers.BatchNormalization()
        self.activation = keras.layers.Activation('relu')

    def call(self, inputs):
        out = self.fc(inputs)
        if self.check:
            out = self.bn(out)
        if self.activation:
            out = self.activation(out)
        return out

