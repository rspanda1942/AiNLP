# -*- coding: utf-8 -*-
import tensorflow as tf
from common import CnnGLU


class CnnGLUBlock(object):

    def __init__(self,
                 block_size = 5,
                 dropout_rate = 0.1,
                 is_batch_norm = True,
                 is_training = True,
                 pad_format="normal",
                 reuse=tf.AUTO_REUSE,
                 name = "CnnGLUBlock"):

        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.is_batch_norm = is_batch_norm
        self.is_training = is_training

        # For language model, pad the left side to prevent conv from future inputs
        self.pad_format = pad_format
        self.reuse = reuse
        self.name = name

        self.GLUblocks = [CnnGLU(128,
                                 [3, 1],
                                 pad_format = self.pad_format,
                                 dropout_rate = self.dropout_rate,
                                 is_batch_norm = self.is_batch_norm,
                                 is_training = self.is_training,
                                 name = "GLU_block_%d" % i)
                          for i in range(self.block_size)]

    def __call__(self, inputs, inputs_mask):
        """

        Args:
            inputs: N * H * W * C
            inputs_mask: N * H * W * 1

        Returns:

        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            for glu in self.GLUblocks:
                inputs = glu(inputs)
                inputs = tf.multiply(inputs, inputs_mask)

        return inputs
