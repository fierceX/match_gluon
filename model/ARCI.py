from mxnet import gluon, nd
from mxnet.gluon import nn


class ARCI(gluon.HybridBlock):
    def __init__(self, vocab_len, embe_size, prefix=None):
        super(ARCI, self).__init__(prefix=prefix)
        with self.name_scope():
            self.Embed = gluon.nn.Embedding(vocab_len, embe_size)
            self.QConv = gluon.nn.Conv1D(channels=100, kernel_size=3)
            self.DConv = gluon.nn.Conv1D(channels=100, kernel_size=3)
            self.Pool = gluon.nn.MaxPool1D(pool_size=2)
            self.output = gluon.nn.Dense(2)

    def hybrid_forward(self, F, q, d):
        qe = F.transpose(self.Embed(q), axes=(0, 2, 1))
        de = F.transpose(self.Embed(d), axes=(0, 2, 1))
        qp = self.Pool(self.QConv(qe))
        dp = self.Pool(self.DConv(de))
        qdp = F.concat(*[qp, dp], dim=1)
        return self.output(qdp)
