# %%
import torch
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
import numpy as np
# %%
class DRMM(nn.HybridBlock):
    def __init__(self, vocab_len, embed_size, hist_size=30, output_num=2, pad_val=1, hist_type='LCH', prefix=None, params=None):
        super(DRMM, self).__init__(prefix=prefix, params=params)
        self.hist_size = hist_size
        self.pad_val = pad_val
        if hist_type in ['LCH', 'NH']:
            self.hist_type = hist_type
        else:
            raise ValueError('hist_type \''+hist_type+'\' not understood. only \'LCH\',\'NH\' are supported.')
        self.ffw = nn.HybridSequential()
        with self.ffw.name_scope():
            self.ffw.add(nn.Dense(10, activation='tanh', flatten=False))
            self.ffw.add(nn.Dense(1, activation='tanh', flatten=False))
        self.embedding = nn.Embedding(vocab_len, embed_size)
        self._attention = nn.Dense(1, use_bias=False, flatten=False)
        self.output = nn.Dense(output_num)

    def hist_map(self, F, x1, x2, mask):
        mm = F.batch_dot(x1, F.transpose(x2, (0, 2, 1)))

        norm1 = F.expand_dims(F.norm(x1, axis=2), axis=2)
        norm2 = F.expand_dims(F.norm(x2, axis=2), axis=2)
        n_n = F.batch_dot(norm1, F.transpose(norm2, (0, 2, 1)))
        cosine_distance = mm / (n_n + mask)

        bin_upperbounds = np.linspace(-1, 1, num=self.hist_size)[1:]

        H = []

        for bin_upperbound in bin_upperbounds:
            H.append((cosine_distance < bin_upperbound).sum(axis=-1) + 1)
        H.append(((cosine_distance > 0.999) *
                  (cosine_distance < 1.001)).sum(axis=-1) + 1)
        matching_hist = F.stack(*H, axis=2)

        if self.hist_type == 'NH':
            matching_hist_sum = matching_hist.sum(axis=-1)
            return F.broadcast_div(matching_hist, F.expand_dims(matching_hist_sum, axis=2))

        if self.hist_type == 'LCH':
            return F.log(matching_hist)

    def attention(self, F, x1, mask):
        w = self._attention(x1).squeeze() + mask
        return F.softmax(w)

    def hybrid_forward(self, F, x1, x2):
        embed_left = self.embedding(x1)
        embed_right = self.embedding(x2)
        left_mask = F.where(x1 != self.pad_val, F.zeros_like(x1), float('-inf')
                            * F.ones_like(x1)).astype('float32')

        right_mask = F.where(x2 != self.pad_val, F.zeros_like(x2), float('-inf')
                             * F.ones_like(x2)).astype('float32')

        new_mask = F.broadcast_add(F.expand_dims(
            right_mask, axis=1), F.expand_dims(left_mask, axis=2))

        new_mask = F.where(new_mask == 0, F.zeros_like(new_mask), float('-inf')
                           * F.ones_like(new_mask)).astype('float32')
        hist = self.hist_map(F, embed_left, embed_right, new_mask)
        x = self.ffw(hist).squeeze()
        w = self.attention(F, embed_left, left_mask)
        out = self.output(w * x)
        return out