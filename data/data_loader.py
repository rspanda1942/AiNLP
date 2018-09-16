# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class DataGenerator(object):

    def __init__(self, data_size):

        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __call__(self):
        for idx in xrange(self.data_size):
            yield np.array(idx, dtype=np.int32)


class LmDataProcess(object):

    def __init__(self, idx_text_data, data_max_length):

        self.idx_text_data = idx_text_data
        self.data_max_length = data_max_length
        # for <SOS>, <EOS>
        self.data_extra_pad = 2
        self.data_length = self.data_max_length + self.data_extra_pad

    def __call__(self, raw_data_idx):

        raw_data = self.idx_text_data[raw_data_idx]
        raw_data = np.array(raw_data, dtype=np.int32)
        raw_length = raw_data.shape[0]

        input_text_idx = np.zeros((self.data_length), dtype=np.int32)
        input_text_idx[0:raw_length] = raw_data

        input_mask = np.zeros((self.data_length), dtype=np.float32)
        input_mask[0:raw_length] = 1.0

        label = np.zeros((self.data_length), dtype=np.int32)
        label[0:raw_length-1] = raw_data[1:]

        label_mask = np.zeros((self.data_length), dtype=np.float32)
        label_mask[0:raw_length-1] = 1.0

        return (input_text_idx, input_mask, label, label_mask)


class TFLmDataSet(object):

    def __init__(self, idx_text_data, FLAGS):

        self.lmDataProcess = LmDataProcess(idx_text_data, FLAGS.max_sentence_length)
        self.lmDataGenerator = DataGenerator(len(idx_text_data))
        self.data_length = self.lmDataProcess.data_length
        self.batch_size = FLAGS.batch_size
        self.iterator = self.setDataSetIterator(FLAGS)

    def setDataSetIterator(self, FLAGS):

        dataset = tf.data.Dataset.from_generator(self.lmDataGenerator, (tf.int32))

        if FLAGS.is_shuffle:
            dataset = dataset.shuffle(FLAGS.data_shuffle_num)

        if FLAGS.is_training:
            dataset = dataset.repeat(FLAGS.num_epochs)

        dataset = dataset.map(lambda raw_data_idx: tf.py_func(func=self.lmDataProcess,
                                                              inp=[raw_data_idx],
                                                              Tout=[tf.int32,
                                                                    tf.float32,
                                                                    tf.int32,
                                                                    tf.float32]),
                              num_parallel_calls=FLAGS.CPU_COUNT)

        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=FLAGS.drop_remainder)
        #dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))
        dataset = dataset.prefetch(FLAGS.data_prefetch_num)
        iterator = dataset.make_initializable_iterator()

        return iterator

    def getBatch(self):

        value = self.iterator.get_next()
        input_text_idx, input_mask, label, label_mask = value

        input_text_idx.set_shape([self.batch_size, self.data_length])
        input_mask.set_shape([self.batch_size, self.data_length])
        label.set_shape([self.batch_size, self.data_length])
        label_mask.set_shape([self.batch_size, self.data_length])

        return input_text_idx, input_mask, label, label_mask

    def getBatch_multiGPU(self, num_GPU):

        value = self.iterator.get_next()
        input_text_idx, input_mask, label, label_mask = value

        input_text_idx.set_shape([self.batch_size, self.data_length])
        input_mask.set_shape([self.batch_size, self.data_length])
        label.set_shape([self.batch_size, self.data_length])
        label_mask.set_shape([self.batch_size, self.data_length])

        input_text_idx_split = tf.split(input_text_idx, num_GPU)
        input_mask_split = tf.split(input_mask, num_GPU)
        label_split = tf.split(label, num_GPU)
        label_mask_split = tf.split(label_mask, num_GPU)

        data_batch_multi = []
        for i in range(num_GPU):
            data_batch_multi.append([input_text_idx_split[i],
                                     input_mask_split[i],
                                     label_split[i],
                                     label_mask_split[i]])

        return data_batch_multi






