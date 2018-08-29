# -*- coding: utf-8 -*-

from io import open


class LmRawDataGenerator(object):
    
    def __init__(self, source, max_sentence_length, max_data_size=None):
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.max_data_size = max_data_size
        
    def __iter__(self):
        for file_path in self.source:
            with open(file_path, 'r', encoding='utf-8') as fread:
                for line in fread:
                    words = line.replace("\n", "").strip().split()
                    if len(words) <= self.max_sentence_length:
                        yield words
