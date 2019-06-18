# %%
from mxnet.gluon import nn
import pickle
import random
import time

import gluonnlp as nlp
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon.data import SimpleDataset
from tqdm import tqdm

from data.QuoraQP import QuoraQP,QuoraQPTransform
from model.DRMM import DRMM
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
# %%
def split_train_valid(data, n=0.8):
    data.shuffle()
    num = int(len(data) * n)
    return SimpleDataset(data[:num]), SimpleDataset(data[num:])


def Acc(pre, label):
    return (pre.argmax(axis=1) == label.astype('float32')).mean().asscalar()


def evaluate(net, dataloader, loss):
    loss_ = 0.0
    acc_ = 0.0
    for ii, _data in enumerate(dataloader):
        q1, q2, _, _, lable = _data
        q1 = q1.as_in_context(mx.gpu(0))
        q2 = q2.as_in_context(mx.gpu(0))

        out = net(q1, q2)
        L = loss(out.as_in_context(mx.cpu()), lable).mean()
        loss_ += L.asnumpy()[0]
        acc_ += Acc(out.as_in_context(mx.cpu()), lable)
    return loss_/(ii+1), acc_/(ii+1)
# %%
tokenizer = nlp.data.SpacyTokenizer('en')
data = QuoraQP('./train.csv', tokenizer)
train_data, valid_data = split_train_valid(data)
vocab = data.build_Vocab()
#%%
embedding = nlp.embedding.create('GloVe', source='glove.840B.300d')
vocab.set_embedding(embedding)
# %%

trans = QuoraQPTransform(vocab, Max_len=50)

train_trans = train_data.transform(trans)
valid_trans = valid_data.transform(trans)

train_dataloader = gluon.data.DataLoader(
    train_trans, batch_size=32, shuffle=True)
valid_dataloader = gluon.data.DataLoader(
    valid_trans, batch_size=128, last_batch='keep')
# %%
net = DRMM(vocab_len=len(vocab), embed_size=300)
net.initialize(init=mx.initializer.Xavier(), ctx=mx.gpu(0))
net.hybridize()
net.embedding.weight.set_data(vocab.embedding.idx_to_vec)
trainer = gluon.Trainer(net.collect_params(), 'Adam')
loss = gluon.loss.SoftmaxCELoss()
net.embedding.params.zero_grad()
# %%
n = 200
for i in range(5):
    loss_ = 0.0
    acc_ = 0.0
    for ii, _data in enumerate(train_dataloader):
        q1, q2, _ ,_, lable = _data
        q1 = q1.as_in_context(mx.gpu(0))
        q2 = q2.as_in_context(mx.gpu(0))

        with autograd.record():
            out = net(q1, q2)
            L = loss(out.as_in_context(mx.cpu()), lable).mean()
        nd.waitall()
        L.backward()
        trainer.update(1)
        loss_ += L.asnumpy()[0]
        acc_ += Acc(out.as_in_context(mx.cpu()), lable)
        if (ii+1) % n == 0:
            print('Epoch: {}, Batch: {}/{}, Loss={:.4f}, Acc={:.4f}'.
                  format(i, ii+1, len(train_dataloader), loss_/n, acc_/n))
            loss_ = 0.0
            acc_ = 0.0
    test_loss, test_acc = evaluate(net, valid_dataloader, loss)
    print('Evaluate Loss={:.4f}, Acc={:.4f}'.format(test_loss, test_acc))

#%%
