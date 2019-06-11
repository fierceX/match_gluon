# %%
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
import numpy as np


class ELU(nn.HybridBlock):
    def __init__(self, alpha=1.0, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self._alpha = alpha

    def hybrid_forward(self, F, x):
        return F.LeakyReLU(x, act_type='elu', slope=self._alpha)


class ESIM(nn.HybridBlock):
    def __init__(self, vocab_len=2000, embed_size=300, hidden_size=300, linear_size=200, output_num=2, dropout=.5, pad_val=1):
        super(ESIM, self).__init__()

        self.pad_val = pad_val
        self.dropout = dropout
        self.Dropout = nn.Dropout(self.dropout)
        self.hidden_size = hidden_size
        self.embeds_dim = embed_size
        self.embedq = nn.Embedding(vocab_len, self.embeds_dim)
        self.embedd = nn.Embedding(vocab_len, self.embeds_dim)

        self.lstm1q = gluon.rnn.LSTM(
            self.hidden_size, bidirectional=True, layout='NTC')
        self.lstm1d = gluon.rnn.LSTM(
            self.hidden_size, bidirectional=True, layout='NTC')

        self.lstm2q = gluon.rnn.LSTM(
            self.hidden_size, bidirectional=True, layout='NTC')
        self.lstm2d = gluon.rnn.LSTM(
            self.hidden_size, bidirectional=True, layout='NTC')

        self.fc = nn.HybridSequential()
        with self.fc.name_scope():
            self.fc.add(nn.Dropout(self.dropout))
            self.fc.add(nn.Dense(linear_size))
            self.fc.add(ELU())
            self.fc.add(nn.Dropout(self.dropout))
            self.fc.add(nn.Dense(output_num))

    def soft_attention_align(self, F, x1, x2, mask1, mask2):
        attention = F.linalg.gemm2(x1, x2, transpose_b=True)
        weight1 = F.softmax(F.broadcast_add(
            attention, F.expand_dims(mask2, axis=1)))
        x1_align = F.linalg.gemm2(weight1, x2)
        weight2 = F.softmax(F.broadcast_add(F.transpose(
            attention, (0, 2, 1)), F.expand_dims(mask1, axis=1)))
        x2_align = F.linalg.gemm2(weight2, x1)
        return x1_align, x2_align

    def submul(self, F, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return F.concat(*[sub, mul], dim=-1)

    def apply_multiple(self, F, x):
        p1 = F.mean(F.transpose(x, (0, 2, 1)), axis=2)
        p2 = F.max(F.transpose(x, (0, 2, 1)), axis=2)
        return F.concat(*[p1, p2], dim=1)

    def hybrid_forward(self, F, x1, x2):

        mask1 = F.where(x1 != self.pad_val, F.zeros_like(x1), float('-inf')
                        * F.ones_like(x1)).astype('float32')
        mask2 = F.where(x2 != self.pad_val, F.zeros_like(x2), float('-inf')
                        * F.ones_like(x2)).astype('float32')

        x1 = F.transpose(self.Dropout(F.transpose(
            self.embedd(x1), (0, 2, 1))), (0, 2, 1))
        x2 = F.transpose(self.Dropout(F.transpose(
            self.embedq(x2), (0, 2, 1))), (0, 2, 1))

        o1 = self.lstm1d(x1)
        o2 = self.lstm1q(x2)

        x1_align, x2_align = self.soft_attention_align(F, o1, o2, mask1, mask2)

        x1_combined = F.concat(
            *[o1, x1_align, self.submul(F, o1, x1_align)], dim=-1)
        x2_combined = F.concat(
            *[o2, x2_align, self.submul(F, o2, x2_align)], dim=-1)

        x1_combined = self.lstm2d(x1_combined)
        x2_combined = self.lstm2q(x2_combined)

        x1_rep = self.apply_multiple(F, x1_combined)
        x2_rep = self.apply_multiple(F, x2_combined)

        x = F.concat(*[x1_rep, x2_rep], dim=-1)
        similarity = self.fc(x)
        return similarity