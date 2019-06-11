import gluonnlp as nlp
import pickle
dic = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}


def build_sequence(filepath1, filepath2):
    with open(filepath1) as f1, open(filepath2, 'w') as f2:
        next(f1)  # skip the header row
        f2.write('\x01'.join(['sent1', 'sent2', 'label'])+'\n')
        for line in f1:
            new_sents = []
            sents = line.strip().split('\t')
            if sents[0] is '-':
                continue

            words_in = sents[1].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(', ')')]
            new_sents.append(' '.join(words_in))

            words_in = sents[2].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(', ')')]
            new_sents.append(' '.join(words_in))

            new_sents.append(dic[sents[0]])

            f2.write('\x01'.join(new_sents) + '\n')

if __name__ == 'main':
    build_sequence('./snli_1.0_train.txt', './new_snli_1.0_train.txt')
    build_sequence('./snli_1.0_test.txt', './new_snli_1.0_test.txt')
    build_sequence('./snli_1.0_dev.txt', './new_snli_1.0_dev.txt')
