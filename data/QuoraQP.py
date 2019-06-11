import gluonnlp as nlp
import pandas as pd
import random
from mxnet.gluon.data import ArrayDataset
from tqdm import tqdm
import numpy as np
import itertools


class QuoraQPTransform(object):
    def __init__(self, vocab, Max_len=20, pad_val=1):
        self.vocab = vocab
        self.Pad = nlp.data.PadSequence(Max_len, pad_val=pad_val)

    def __call__(self, *record):
        _id = record[0]
        q1 = record[3]
        q2 = record[4]
        label = record[5]
        q1 = self.vocab.to_indices(q1)
        q2 = self.vocab.to_indices(q2)

        return self.Pad(np.array(q1, dtype='int32')), self.Pad(np.array(q2, dtype='int32')), np.array(len(q1)), np.array(len(q2)), np.array(label)


class QuoraQP(ArrayDataset):
    def __init__(self, file_path, tokenizer, is_train=True):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.is_train = is_train
        super(QuoraQP, self).__init__(self._read_data())

    def _read_data(self):
        records = []
        data = pd.read_csv(self.file_path)
        data = data.fillna('')
        for i in tqdm(range(len(data))):
            d = data.iloc[i]
            question1 = self.tokenizer(d['question1'])
            question2 = self.tokenizer(d['question2'])
            if len(question1) > 0 and len(question2) > 0:
                if self.is_train:
                    record = (d['id'], d['qid1'], d['qid2'],
                              (question1), (question2), d['is_duplicate'])
                else:
                    record = (d['test_id'], '', '',
                              (question1), (question2), '')
                records.append(record)
        return records

    def shuffle(self):
        random.shuffle(self._data[0])

    def build_Vocab(self):
        train_seqs = []
        for sample in self._data[0]:
            train_seqs.append(sample[3])
            train_seqs.append(sample[4])

        counter = nlp.data.count_tokens(
            list(itertools.chain.from_iterable(train_seqs)))

        vocab = nlp.Vocab(counter)
        return vocab
