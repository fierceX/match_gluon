from mxnet import gluon, nd
from mxnet.gluon import nn


class ARCII(gluon.HybridBlock):
    def __init__(self, vocab_len, embe_size, prefix=None):
        super(ARCII, self).__init__(prefix=prefix)
        with self.name_scope():
            self.Embed = gluon.nn.Embedding(vocab_len, embe_size)
            self.QConv1d = gluon.nn.Conv1D(
                channels=32, kernel_size=3, padding=1)
            self.DConv1d = gluon.nn.Conv1D(
                channels=32, kernel_size=3, padding=1)
            self._conv_pool_block = gluon.nn.HybridSequential()
            with self._conv_pool_block.name_scope():
                self._conv_pool_block.add(gluon.nn.Conv2D(
                    channels=64, kernel_size=3, activation='relu', padding=(1, 1)))
                self._conv_pool_block.add(gluon.nn.MaxPool2D(pool_size=3))
                self._conv_pool_block.add(gluon.nn.Conv2D(
                    channels=64, kernel_size=3, activation='relu', padding=(1, 1)))
                self._conv_pool_block.add(gluon.nn.MaxPool2D(pool_size=3))
            self.out_layer = gluon.nn.HybridSequential()
            with self.out_layer.name_scope():
                self.out_layer.add(gluon.nn.Dropout(.5))
                self.out_layer.add(gluon.nn.Dense(2))

    def hybrid_forward(self, F, q, d):
        qe = F.transpose(self.Embed(q), axes=(0, 2, 1))
        de = F.transpose(self.Embed(d), axes=(0, 2, 1))
        qc1 = self.QConv1d(qe)
        dc1 = self.DConv1d(de)
        dc1_ex = F.stack(*[dc1] * qc1.shape[2], axis=2)
        qc1_ex = F.stack(*[qc1] * dc1.shape[2], axis=3)
        qdc1 = qc1_ex + dc1_ex
        qdc2 = self._conv_pool_block(qdc1)
        out = self.out_layer(qdc2)
        return out
