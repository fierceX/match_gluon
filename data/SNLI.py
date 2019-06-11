import gluonnlp as nlp
import pandas as pd
import random
from mxnet.gluon.data import ArrayDataset
from tqdm import tqdm
import numpy as np
import itertools

class SNLIData(ArrayDataset):
    def __init__(self, file_path,max_len=100, tokenizer=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        super(SNLIData, self).__init__(self._read_data())

    def _read_data(self):
        records = []
        examples_len = []
        with open(self.file_path, 'r') as f:
            next(f)
            for i, d in enumerate(f):
                d = d.strip().split('\x01')
                if self.tokenizer is not None:
                    sent1 = self.tokenizer(d[0])
                    sent2 = self.tokenizer(d[0])
                else:
                    sent1 = d[0].split()
                    sent2 = d[1].split()
                sent1 = sent1 if len(sent1) > self.max_len else sent1[:self.max_len]
                sent2 = sent2 if len(sent2) > self.max_len else sent2[:self.max_len]
                label = d[2]
                examples_len.append((len(sent1),len(sent2)))
                records.append((i, sent1, sent2, label))
        self.data_len = examples_len
        return records

    def build_Vocab(self):
        train_seqs = []
        for sample in self._data[0]:
            train_seqs.append(sample[1])
            train_seqs.append(sample[2])

        counter = nlp.data.count_tokens(
            list(itertools.chain.from_iterable(train_seqs)))

        vocab = nlp.Vocab(counter)
        return vocab


class SNLIPTransform(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, *record):
        _id = record[0]
        q1 = record[1]
        q2 = record[2]
        label = record[3]
        q1 = self.vocab.to_indices(q1)
        q2 = self.vocab.to_indices(q2)

        return np.array(q1, dtype='int32'), np.array(q2, dtype='int32'), np.array(len(q1), dtype='int32'), np.array(len(q2), dtype='int32'), np.array(label, dtype='int32')
