# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
from decoder import Decoder
from layers import Embedding


class SampledSoftmaxDecoder(object):

    def __init__(self,
                 FLAGS,
                 init=tf.contrib.layers.xavier_initializer(uniform=False)):
        self.vocab_size = FLAGS.vocab_size
        self.embedding_dim = FLAGS.embedding_dim
        self.sampled_softmax_num = FLAGS.sampled_softmax_num
        self.init = init

    def dense_decode(self,
                     inputs,
                     labels_sequence_length,
                     labels):

        """

        Args:
            inputs: B * W * D
            sequence_length: B
            labels: B * W

        Returns:

        """

        # share the embedding with encoder
        with tf.variable_scope('wordEmbedding', reuse=tf.AUTO_REUSE):
            wordEmbedding = tf.get_variable('wordEmbedding',
                                            shape=[self.vocab_size, self.embedding_dim],
                                            initializer=self.init,
                                            dtype=tf.float32,
                                            trainable=True)

        bias = tf.get_variable("decoder_bias", [self.vocab_size], initializer=tf.constant_initializer(0.0))

        batch_size = tf.shape(inputs)[0]
        sentence_length = tf.shape(inputs)[1]
        dims = tf.shape(inputs)[2]

        inputs = tf.reshape(inputs, [-1, dims])
        labels = tf.reshape(labels, [-1, 1])

        loss = tf.nn.sampled_softmax_loss(wordEmbedding,
                                          bias,
                                          labels,
                                          inputs,
                                          self.sampled_softmax_num,
                                          self.vocab_size,
                                          partition_strategy="div")

        loss = tf.reshape(loss, [batch_size, sentence_length])

        inputs_mask = tf.sequence_mask(
            labels_sequence_length, maxlen=sentence_length, dtype=tf.float32)

        loss = tf.multiply(loss, inputs_mask)

        return loss
