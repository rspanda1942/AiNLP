# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from model import Model
import tensorflow as tf


class CnnGLULanguageModel(Model):

    def __init__(self,
                 encoder,
                 decoder,
                 share_embedding=None,
                 name="CnnGLULanguageModel"):
        """
        Args:
            encoder:
            decoder:
            share_embedding:
            name:
        """

        super(CnnGLULanguageModel, self).__init__(name)

        self.encoder = encoder
        self.decoder = decoder
        self.share_embedding = share_embedding

    def _build(self, inputs, vocab, FLAGS):

        input_text_idx, input_mask, label, label_mask = inputs
        sequence_length = tf.reduce_sum(input_mask, 1)
        label_sequence_length = tf.reduce_sum(label_mask, 1)

        with tf.variable_scope(self.name):
            encoder_output, encoder_state, sequence_length = self.encoder.encode(input_text_idx, sequence_length)
            loss = self.decoder.dense_decode(encoder_output, label_sequence_length, label)
            loss = tf.reduce_sum(loss, 1) / label_sequence_length
            loss = tf.reduce_mean(loss)

        return loss
