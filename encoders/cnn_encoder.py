# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
from encoder import Encoder
from layers import Embedding, CnnGLUBlock


class CNNGLUEncoder(Encoder):

    def __init__(self, FLAGS):
        self.embeddingLayers = Embedding(FLAGS.vocab_size, FLAGS.embedding_dim)
        self.cnnGLUBlock = CnnGLUBlock(dropout_rate = FLAGS.dropout_rate,
                                       is_batch_norm = FLAGS.is_batch_norm,
                                       is_training = FLAGS.is_training,
                                       pad_format = FLAGS.pad_format)

    def encode(self, inputs, sequence_length=None, mode=None):

        max_sentence_length = tf.shape(inputs)[1]

        inputs_mask = tf.sequence_mask(
            sequence_length, maxlen=max_sentence_length, dtype=tf.float32)
        inputs_mask = tf.expand_dims(inputs_mask, 2)
        inputs_mask = tf.expand_dims(inputs_mask, 3)

        embedding = self.embeddingLayers(inputs)
        embedding = tf.expand_dims(embedding, 2)
        encoder_output = self.cnnGLUBlock(embedding, inputs_mask)

        encoder_output = tf.squeeze(encoder_output, 2)
        encoder_state = tf.reduce_mean(encoder_output, axis=1)

        return (encoder_output, encoder_state, sequence_length)
