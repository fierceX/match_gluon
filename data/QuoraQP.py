import gluonnlp as nlp
import pandas as pd
from mxnet.gluon.data import ArrayDataset
from tqdm import tqdm


class QuoraQP(ArrayDataset):
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer
        super(QuoraQP, self).__init__(self._read_data())

    def _read_data(self):
        records = []
        data = pd.read_csv(self.file_path)
        data = data.fillna('')
        for i in tqdm(range(len(data))):
            d = data.iloc[i]
            question1 = self.tokenizer(d['question1'])
            question2 = self.tokenizer(d['question2'])
            record = (d['id'], d['qid1'], d['qid2'],
                      (question1), (question2), d['is_duplicate'])
            records.append(record)
        return records
