import gluonnlp as nlp
import pandas as pd
import random
from mxnet.gluon.data import ArrayDataset
from tqdm import tqdm


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
            if self.is_train:
                record = (d['id'], d['qid1'], d['qid2'],
                          (question1), (question2), d['is_duplicate'])
            else:
                record = (d['test_id'], '', '', (question1), (question2), '')
            records.append(record)
        return records

    def shuffle(self):
        random.shuffle(self._data[0])
