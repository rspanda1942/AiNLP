# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import constants
import tensorflow as tf
import numpy as np


class Conv2d(object):

    def __init__(self,
                 filter_size=128,
                 kernel_size=[3, 3],
                 strides=[1, 1, 1, 1],
                 dilations=[1, 1, 1, 1],
                 data_format="NHWC",
                 kernel_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                 reuse = None,
                 is_weight_norm = False,
                 name="Conv2d"):

        """

        Args:
            inputs: N * H * W * C
            filter_size:
            kernel_size:
            strides:
            dilations:
            padding:
            data_format:
            kernel_init:
            name:

        Returns:

        """
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilations
        self.data_format = data_format
        self.kernel_init = kernel_init
        self.reuse = reuse
        self.is_weight_norm = is_weight_norm
        self.name = name

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            input_dimension = inputs.get_shape().as_list()[3]
            conv_kernel_size = [self.kernel_size[0], self.kernel_size[1], input_dimension, self.filter_size]

            kernel_para = tf.Variable(self.kernel_init(conv_kernel_size), name="kernel_para")
            bias_para = tf.Variable(tf.zeros([self.filter_size]), name="bias_para")

            if self.is_weight_norm:
                # do weight norm
                pass

            inputs = tf.nn.conv2d(inputs,
                                  kernel_para,
                                  self.strides,
                                  "VALID",
                                  data_format = self.data_format,
                                  dilations = self.dilations)
            inputs = tf.nn.bias_add(inputs, bias_para)

        return inputs


class Padding(object):

    def __init__(self,
                 kernel_size,
                 pad_format=constants.PAD_FORMAT_NORMAL,
                 data_format="NHWC"):
        """

        Args:
            kernel_size: [H, W]
            data_format: NHWC
            pad_format: left, normal, right
        """

        self.kernel_size = kernel_size
        self.data_format = data_format
        self.pad_format = pad_format

    def __call__(self, inputs):

        if self.data_format != "NHWC":
            raise RuntimeError

        if self.pad_format == constants.PAD_FORMAT_PREFIX:
            height_left_pad = self.kernel_size[0] - 1
            width_left_pad = self.kernel_size[1] - 1
            inputs = tf.pad(inputs, [[0, 0], [height_left_pad, 0], [width_left_pad, 0], [0, 0]])
        elif self.pad_format == constants.PAD_FORMAT_NORMAL:
            height_pad = int((self.kernel_size[0] - 1)/2)
            width_pad = int((self.kernel_size[1] - 1)/2)
            inputs = tf.pad(inputs, [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]])

        return inputs


class CnnGLU(object):

    def __init__(self,
                 filter_size,
                 kernel_size,
                 strides=[1, 1, 1, 1],
                 dilations=[1, 1, 1, 1],
                 pad_format=constants.PAD_FORMAT_NORMAL,
                 data_format="NHWC",
                 kernel_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                 reuse = None,
                 dropout_rate = 0.1,
                 is_weight_norm = False,
                 is_batch_norm = True,
                 is_residual = True,
                 is_training = True,
                 name="CnnGLU"):

        """

        Args:
            filter_size:
            kernel_size:
            strides:
            dilations:
            padding:
            data_format:
            kernel_init:
            reuse:
            is_weight_norm:
            is_batch_norm:
            is_training:
            name:
        """

        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilations
        self.pad_format = pad_format
        self.data_format = data_format
        self.kernel_init = kernel_init
        self.reuse = reuse
        self.dropout_rate = dropout_rate
        self.is_weight_norm = is_weight_norm
        self.is_batch_norm = is_batch_norm
        self.is_residual = is_residual
        self.is_training = is_training
        self.name = name

        self.padding = Padding(kernel_size, pad_format=self.pad_format)

        self.conv2dLayers = Conv2d(filter_size,
                                   kernel_size,
                                   strides,
                                   dilations,
                                   data_format,
                                   kernel_init,
                                   reuse,
                                   is_weight_norm,
                                   name="conv2d")

        self.conv2dGateLayers = Conv2d(filter_size,
                                       kernel_size,
                                       strides,
                                       dilations,
                                       data_format,
                                       kernel_init,
                                       reuse,
                                       is_weight_norm,
                                       name="conv2d_gate")

    def __call__(self, inputs):

        """

        Args:
            inputs: N * H * W * C

        Returns:

        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            shortcut = inputs

            if self.is_batch_norm:
                inputs = tf.layers.batch_normalization(inputs,
                                                       axis=3,
                                                       fused=True,
                                                       training=self.is_training,
                                                       epsilon=1e-05,
                                                       name="batch_norm")

            inputs = self.padding(inputs)
            inputs_feature = self.conv2dLayers(inputs)
            inputs_gate = self.conv2dGateLayers(inputs)

            inputs = tf.multiply(inputs_feature, tf.sigmoid(inputs_gate))

            if self.is_residual:
                inputs = inputs + shortcut

            inputs = tf.layers.dropout(inputs,
                                       rate=self.dropout_rate,
                                       training=self.is_training)

        return inputs
