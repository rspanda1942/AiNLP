# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class Embedding(object):

    def __init__(self,
                 words_num,
                 embedding_num,
                 is_initWeight = False,
                 initWeight = None,
                 scale = False,
                 init = tf.contrib.layers.xavier_initializer(uniform=False),
                 name = "wordEmbedding"):

        self.words_num = words_num
        self.embedding_num = embedding_num
        self.is_initWeight = is_initWeight
        self.initWeight = initWeight
        self.scale = scale
        self.init = init
        self.name = name

    def __call__(self, inputs):
        """

        Args:
            inputs: A 2d Tensor with shape of (B, W)

        Returns:
            A 3d Tensor with shape of (B, W, D)
        """

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            if self.is_initWeight:
                wordEmbedding = tf.get_variable('wordEmbedding',
                                                initializer=self.initWeight,
                                                dtype=tf.float32,
                                                trainable=True)
            else:
                wordEmbedding = tf.get_variable('wordEmbedding',
                                                shape=[self.words_num, self.embedding_num],
                                                initializer=self.init,
                                                dtype=tf.float32,
                                                trainable=True)

            # idx 0 denote <Padding>, mask the 0 idx
            raw_mask_array = [[0.]] + [[1.]] * (self.words_num - 1)
            mask_padding_lookup_table = tf.get_variable('mask_padding_lookup_table',
                                                        initializer=raw_mask_array,
                                                        dtype=tf.float32,
                                                        trainable=False)

            mask_padding_input = tf.nn.embedding_lookup(mask_padding_lookup_table, inputs)
            embedding = tf.nn.embedding_lookup(wordEmbedding, inputs)
            embedding = tf.multiply(embedding, mask_padding_input)

            if self.scale:
                # https://arxiv.org/abs/1608.05859
                # scale the embedding
                embedding = embedding * (self.embedding_num ** 0.5)

        return embedding


class PositionalEmbedding(object):
    # https://github.com/Kyubyong/transformer
    def __init__(self,
                 num_units,
                 max_position = 512,
                 trainable = True,
                 scale = False,
                 name = "positionalEmbedding"):

        self.num_units = num_units
        self.max_position = max_position
        self.trainable = trainable
        self.scale = scale
        self.name = name

    def __call__(self, inputs, inputs_mask=None):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            batch_size, sentence_length = tf.shape(inputs)
            position_ind = tf.tile(tf.expand_dims(tf.range(sentence_length), 0), [batch_size, 1])

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, 2. * i / self.num_units) for i in range(self.num_units)]
                for pos in range(self.max_position)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

            lookup_table = tf.get_variable('position_lookup_table',
                                            initializer=position_enc,
                                            dtype=tf.float32,
                                            trainable=self.trainable)

            position_outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
            if inputs_mask is not None:
                position_outputs = tf.multiply(position_outputs, tf.expand_dims(inputs_mask, 2))

            if self.scale:
                # https://arxiv.org/abs/1608.05859
                # scale the embedding
                position_outputs = position_outputs * (self.num_units ** 0.5)

            return position_outputs
