# -*- coding: utf-8 -*-
import argparse
import sys
import time
import constants

parser = argparse.ArgumentParser(description='Language model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train_data', type=str,
                    default='./data_sample/language_model/train_samples.txt',
                    # default='/home/panda/data/chinese_corpus/wiki/wiki_seg.txt',
                    help='Training file.'
                         'E.g., single file /data/train.txt'
                         'multiple files /data/train.txt,/data/train.txt')

parser.add_argument('--save_path', type=str,
                    default='./temp/')

parser.set_defaults(
    network = 'GluCNN',

    batch_size = 128,
    drop_remainder = True,
    max_sentence_length = 128,
    wordCutNum = 1,
    vocab_size = 0,

    embedding_dim = 128,
    dropout_rate = 0.15,
    learning_rate = 0.001,

    num_epochs = 32,
    num_steps = 0,

    sampled_softmax_num = 256,

    GPU_device_ids = '0',

    data_size = 0,
    data_prefetch_num = 1024,
    data_shuffle_num = 0,
    mode = constants.TRAIN_KEY,
    is_batch_norm=True,
    is_training=True,
    pad_format=constants.PAD_FORMAT_PREFIX,
    is_shuffle = True,
    save_flag = '20180916_1616',

    watchnum = 1
)


# FLAGS = parser.parse_args(sys.argv[3:])
FLAGS = parser.parse_args()


import os
import multiprocessing
from utils.common import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_device_ids

# Performance Improvement#
# 1. Auto-tune
# This is no longer needed with recent versions of TF
# And actually seems to make the performance worse
# os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = "1"

import tensorflow as tf
import numpy as np

print("OS: ""OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)
print("GPU: ", get_gpu_name())
print(get_cuda_version())
print("CuDNN Version ", get_cudnn_version())

CPU_COUNT = multiprocessing.cpu_count()
GPU_COUNT = len(get_gpu_name())
print("CPUs: ", CPU_COUNT)
print("GPUs: ", GPU_COUNT)

FLAGS.CPU_COUNT = CPU_COUNT
FLAGS.GPU_COUNT = len(FLAGS.GPU_device_ids.split(","))

from data import LmRawDataGenerator, LmVocabulary

data_raw = LmRawDataGenerator(FLAGS.train_data.split(","),
               max_sentence_length = FLAGS.max_sentence_length)

vocab = LmVocabulary(data_raw, FLAGS.wordCutNum)
vocab.save_vocab(FLAGS)
FLAGS.vocab_size = vocab.vocab_size

idx_data = vocab.idx_data_gen(data_raw)
FLAGS.num_steps = int(1.0*len(idx_data) / FLAGS.batch_size)
FLAGS.data_size = int(FLAGS.num_steps * FLAGS.batch_size)
FLAGS.data_shuffle_num = len(idx_data)
print "data_size: ", FLAGS.data_size

for arg in vars(FLAGS):
    print arg, getattr(FLAGS, arg)

from data import TFLmDataSet

tfLmDataset = TFLmDataSet(idx_data, FLAGS)
data_batch = tfLmDataset.getBatch()

from encoders import CNNGLUEncoder
from decoders import SampledSoftmaxDecoder
from models import CnnGLULanguageModel

cnnGLUEncoder = CNNGLUEncoder(FLAGS)
sampledSoftmaxDecoder = SampledSoftmaxDecoder(FLAGS)

languageModel = CnnGLULanguageModel(cnnGLUEncoder,
                                    sampledSoftmaxDecoder)


loss = languageModel(data_batch, vocab, FLAGS)

g_list = tf.global_variables()
bn_para = [g for g in g_list if "batch_norm/moving" in g.name]
extra_para = [g for g in g_list if "mask_padding_lookup_table" in g.name]

global_steps = tf.Variable(0, trainable=False)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss, global_step=global_steps)

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

running_loss = 0.0
running_time = 0.0

save_para_list = tf.trainable_variables() + bn_para + extra_para
saver = tf.train.Saver(save_para_list)

with tf.Session(config=run_config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tfLmDataset.iterator.initializer)

    while True:

        train_op = [loss, global_steps, optim]
        start_time = time.time()
        try:
            loss_list = sess.run(train_op)

        except tf.errors.OutOfRangeError:
            print "finish training"
            break

        running_loss += loss_list[0]
        global_count = loss_list[1]
        running_time += time.time() - start_time

        if global_count % FLAGS.watchnum == 0:
            print('[%d, %5d] lr: %.6f, Perplexity: %.3f, time: %.3f s/batch' %
                  (int(1.0*global_count/FLAGS.num_steps),
                   global_count % FLAGS.num_steps,
                   FLAGS.learning_rate,
                   np.exp(running_loss/FLAGS.watchnum),
                   running_time/FLAGS.watchnum))
            running_loss = 0.0
            running_time = 0.0

    save_path = saver.save(sess, FLAGS.save_path + FLAGS.save_flag + "/GLUCNN_LangageModel_final.ckpt")
    print("Model saved in path: %s" % save_path)
