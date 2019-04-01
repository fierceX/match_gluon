import itertools

import gluonnlp as nlp
import numpy as np


class MatchTransform(object):
    def __init__(self, vocab, Max_len=20):
        self.vocab = vocab
        self.Pad = nlp.data.PadSequence(Max_len, pad_val=0)

    def __call__(self, *record):
        _id = record[0]
        q1 = record[3]
        q2 = record[4]
        label = record[5]
        q1 = self.vocab.to_indices(q1)
        q2 = self.vocab.to_indices(q2)

        return self.Pad(np.array(q1, dtype='int32')), self.Pad(np.array(q2, dtype='int32')), np.array(label)


def build_Vocab(data):
    train_seqs = []
    for sample in data:
        train_seqs.append(sample[3])
        train_seqs.append(sample[4])

    counter = nlp.data.count_tokens(
        list(itertools.chain.from_iterable(train_seqs)))

    vocab = nlp.Vocab(counter)
    return vocab
