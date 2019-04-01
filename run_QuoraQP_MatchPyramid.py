# %%
import time

import gluonnlp as nlp
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from tqdm import tqdm

from data.QuoraQP import QuoraQP
from data.util import MatchTransform, build_Vocab
from model.MatchPyramid import MatchPyramid

# %%

tokenizer = nlp.data.SpacyTokenizer('en')
data = QuoraQP('./train.csv', tokenizer)
# %%
vocab = build_Vocab(data)
# %%
trans = MatchTransform(vocab, Max_len=50)
data_trans = data.transform(trans)
dataloader = gluon.data.DataLoader(data_trans, batch_size=64, shuffle=True)
# %%
net = MatchPyramid(vocab_len=28165, embe_size=300, reft_len=20, right_len=20)
net.initialize(ctx=mx.gpu(0))
trainer = gluon.Trainer(net.collect_params(), 'Adam')
loss = gluon.loss.SoftmaxCELoss()
loss_ = 0.0
n = 200
for i in range(5):
    for ii, _data in enumerate(dataloader):
        q1, q2, lable = _data
        q1 = q1.as_in_context(mx.gpu(0))
        q2 = q2.as_in_context(mx.gpu(0))
        with autograd.record():
            out = net(q1, q2)
            L = loss(out.as_in_context(mx.cpu()), lable).mean()
        L.backward()
        nd.waitall()
        trainer.step(1)
        loss_ += L.asnumpy()[0]
        if ii % n == 0:
            print('Epoch: {}, Batch: {}/{}, Loss={:.4f}'.
                  format(i, ii, len(dataloader), loss_/n))
            loss_ = 0.0
