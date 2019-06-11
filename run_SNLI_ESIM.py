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

from data.SNLI import SNLIData, SNLIPTransform
from model.ESIM import ESIM
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
# %%


def Acc(pre, label):
    return (pre.argmax(axis=1) == label.astype('float32')).mean().asscalar()


def evaluate(net, dataloader, loss):
    loss_ = 0.0
    acc_ = 0.0
    for ii, _data in enumerate(dataloader):
        q1, q2, q1_len, q2_len, lable = _data
        q1 = q1.as_in_context(mx.gpu(0))
        q2 = q2.as_in_context(mx.gpu(0))
        q1_len = q1_len.as_in_context(mx.gpu(0))
        q2_len = q2_len.as_in_context(mx.gpu(0))

        out = net(q1, q2)
        L = loss(out.as_in_context(mx.cpu()), lable).mean()
        loss_ += L.asnumpy()[0]
        acc_ += Acc(out.as_in_context(mx.cpu()), lable)
    return loss_/(ii+1), acc_/(ii+1)


# %%
# tokenizer = nlp.data.SpacyTokenizer('en')
train_data = SNLIData('./data/new_snli_1.0_train.txt')
dev_data = SNLIData('./data/new_snli_1.0_dev.txt')
test_data = SNLIData('./data/new_snli_1.0_test.txt')
vocab = train_data.build_Vocab()
# %%

trans = SNLIPTransform(vocab)

train_trans = train_data.transform(trans)
valid_trans = dev_data.transform(trans)
test_trans = test_data.transform(trans)

batch_sampler = nlp.data.sampler.FixedBucketSampler(
    train_data.data_len, batch_size=32, num_buckets=10, ratio=0, shuffle=True)
print(batch_sampler.stats())
# %%

batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Pad(axis=0, pad_val=1),
    nlp.data.batchify.Pad(axis=0, pad_val=1),
    nlp.data.batchify.Stack(),
    nlp.data.batchify.Stack(),
    nlp.data.batchify.Stack()
)

train_dataloader = mx.gluon.data.DataLoader(
    train_trans, batch_sampler=batch_sampler, batchify_fn=batchify_fn, num_workers=4)

valid_dataloader = gluon.data.DataLoader(
    valid_trans, batchify_fn=batchify_fn, batch_size=128, last_batch='keep')

test_dataloader = gluon.data.DataLoader(
    test_trans, batchify_fn=batchify_fn, batch_size=128, last_batch='keep')

# %%
net = ESIM(vocab_len=len(vocab), embed_size=300,
           hidden_size=300, linear_size=300, output_num=3)
net.initialize(init=mx.initializer.Xavier(), ctx=mx.gpu(0))
net.hybridize()
# %%
embedding = nlp.embedding.create('GloVe', source='glove.840B.300d')
vocab.set_embedding(embedding)
net.embedq.weight.set_data(vocab.embedding.idx_to_vec)
net.embedd.weight.set_data(vocab.embedding.idx_to_vec)
# %%
trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate': 4e-4})
loss = gluon.loss.SoftmaxCELoss()
# %%
n = 200
for i in range(15):
    loss_ = 0.0
    acc_ = 0.0
    for ii, _data in enumerate(train_dataloader):
        q1, q2, q1_len, q2_len, lable = _data
        q1 = q1.as_in_context(mx.gpu(0))
        q2 = q2.as_in_context(mx.gpu(0))

        q1_len = q1_len.as_in_context(mx.gpu(0))
        q2_len = q2_len.as_in_context(mx.gpu(0))
        with autograd.record():
            out = net(q1, q2)
            L = loss(out.as_in_context(mx.cpu()), lable).mean()
        nd.waitall()
        L.backward()
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(net.collect_params().values(), 1)
        trainer.update(1)
        loss_ += L.asnumpy()[0]
        acc_ += Acc(out.as_in_context(mx.cpu()), lable)
        if (ii+1) % n == 0:
            print('Epoch: {}, Batch: {}/{}, Loss={:.4f}, Acc={:.4f}'.
                  format(i, ii+1, len(train_dataloader), loss_/n, acc_/n))
            loss_ = 0.0
            acc_ = 0.0
    dev_loss, dev_acc = evaluate(net, valid_dataloader, loss)
    test_loss, test_acc = evaluate(net, test_dataloader, loss)
    print('Evaluate dev Loss={:.4f}, Acc={:.4f}, test Loss={:.4f}, Acc={:.4f}'.format(
        dev_loss, dev_acc, test_loss, test_acc))

# %%
