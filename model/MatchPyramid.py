import math

from mxnet import gluon, nd
from mxnet.gluon import nn


class MatchPyramid(nn.Block):
    def __init__(self, vocab_len, reft_len, right_len, embe_size=300, psize=[10, 10], kernel_num=[3, 5], kernel_size=[3, 3], prefix=None):
        super(MatchPyramid, self).__init__(prefix=prefix)
        self.embed_left = nn.Embedding(vocab_len, embe_size)
        self.embed_right = nn.Embedding(vocab_len, embe_size)
        self._conv_block = nn.HybridSequential()
        with self._conv_block.name_scope():
            self._conv_block.add(
                nn.Conv2D(channels=kernel_num[0], kernel_size=kernel_size[0], padding=1))
            self._conv_block.add(
                nn.Conv2D(channels=kernel_num[1], kernel_size=kernel_size[1], padding=1))
        stride1 = math.ceil(reft_len / psize[0])
        stride2 = math.ceil(right_len / psize[1])
        self.pool = nn.MaxPool2D(pool_size=(stride1, stride2), strides=(
            stride1, stride2), ceil_mode=True)

        self.output_layer = nn.HybridSequential()
        with self.output_layer.name_scope():
            self.output_layer.add(nn.Dropout(.5))
            self.output_layer.add(nn.Dense(2))

    def forward(self, x_left, x_right):
        x_left = self.embed_left(x_left)
        x_right = self.embed_right(x_right)
        embed_cross = nd.expand_dims(nd.batch_dot(
            x_left, x_right, transpose_b=True), 3)
        embed_cross = nd.transpose(embed_cross, (0, 3, 1, 2))
        embed_cross = self._conv_block(embed_cross)
        embed_pool = self.pool(embed_cross)
        out = self.output_layer(embed_pool)
        return out
