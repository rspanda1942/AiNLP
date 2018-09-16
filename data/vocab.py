# -*- coding: utf-8 -*-
from collections import Counter
import os
from io import open
import time


class LmVocabulary(object):

    def __init__(self, rawDataGenerator, wordCutNum, restore_path=None):
        self.wordCutNum = wordCutNum
        self._pad = 0
        self._unk = 1
        self._bos = 2
        self._eos = 3
        self.corpus_size = 0
        self.vocab_size = 0

        if restore_path is not None:
            pass
        else:
            self._word_to_id, self._id_to_word = self.vocab_gen(rawDataGenerator)

    def vocab_gen(self, rawDataGenerator):
        vocab = Counter()
        for no, words in enumerate(rawDataGenerator):
            for word in words:
                vocab[word] += 1
            self.corpus_size += 1
            if no % 1000000 == 0:
                print("process %d sentence, get %d vocab" % (no, len(vocab)))
        print("process %d sentence, get %d vocab" % (no, len(vocab)))

        vocab_cut = {k: v for k, v in vocab.items() if v >= self.wordCutNum}
        vocab_sorted = sorted(vocab_cut.items(), key=lambda x: x[1], reverse=True)
        vocab_post = {k: idx+4 for idx, (k, _) in enumerate(vocab_sorted)}

        vocab_post[u'<Padding>'] = 0
        vocab_post[u'<UNK>'] = 1
        vocab_post[u'<SOS>'] = 2
        vocab_post[u'<EOS>'] = 3

        word, idx = zip(*vocab_post.items())
        idx2word = dict(zip(idx, word))

        self.vocab_size = len(vocab_post)
        print("process %d sentence, get %d vocab_post" % (no, self.vocab_size))
        return vocab_post, idx2word

    def save_vocab(self, FLAGS):

        my_time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
        if FLAGS.save_flag == '':
            FLAGS.save_flag = my_time
        else:
            pass

        vocab_save_path = FLAGS.save_path + "/" + FLAGS.save_flag + "/"
        if os.path.exists(vocab_save_path):
            pass
        else:
            os.mkdir(vocab_save_path)

        vocab_save_path = vocab_save_path + 'vocab_' + str(self.wordCutNum) + '.txt'
        with open(vocab_save_path, 'w', encoding='utf-8') as fwrite:
            for k, v in self._word_to_id.items():
                fwrite.write(k + "\t" + str(v) + "\n")

    def word_to_id(self, word):
        try:
            return self._word_to_id[word]
        except KeyError:
            return self._word_to_id[u'<UNK>']

    def id_to_word(self, idx):
        return self._id_to_word[idx]

    def encode(self, sentence):
        idx_data = [self._word_to_id[u'<SOS>']]
        for word in sentence:
            idx_data.append(self.word_to_id(word))
        idx_data.append(self._word_to_id[u'<EOS>'])
        return idx_data

    def decode(self, idx_data):
        return ' '.join([self.id_to_word(idx) for idx in idx_data])

    def idx_data_gen(self, rawDataGenerator):
        idx_data = []
        for no, raw_data in enumerate(rawDataGenerator):
            idx_data.append(self.encode(raw_data))
            if no % 1000000 == 0:
                print("process %d sentence" % no)

        return idx_data
