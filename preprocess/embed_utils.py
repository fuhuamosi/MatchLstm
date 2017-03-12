# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from preprocess.file_utils import serialize, deserialize, DATA_DIR
import os
from gensim.models import Word2Vec

CORPUS_PATH = os.path.join(DATA_DIR, 'corpus.txt')
EMBEDDING_SIZE = 300
WINDOW_SIZE = 5
MIN_CNT = 5
_PAD = '_PAD'
_NULL = '_NULL'
PAD_ID = 0
NULL_ID = 1


def prepare_corpus():
    """
    将训练集的句子去重后用做训练词向量的语料库
    :return:
    """
    train_path = os.path.join(DATA_DIR, 'snli_1.0_train.bin')
    train_data = deserialize(train_path)
    sents = []
    for d in train_data:
        for word_list in d[:2]:
            sent = ' '.join(word_list)
            sents.append(sent)
    sents = set(sents)
    with open(CORPUS_PATH, 'w') as f:
        for s in sents:
            f.write(s + '\n')
        print('writing {} sents to {}'.format(len(sents), CORPUS_PATH))


def read_corpus(filename):
    corpus = []
    with open(filename, 'r') as datafile:
        for line in datafile:
            corpus.append(line.strip().split(' '))
    return corpus


def main():
    prepare_corpus()
    corpus = read_corpus(CORPUS_PATH)
    model = Word2Vec(corpus, size=EMBEDDING_SIZE, window=WINDOW_SIZE,
                     min_count=MIN_CNT, workers=4)
    index2word = [_PAD, _NULL] + model.wv.index2word
    word2index = dict([(y, x) for (x, y) in enumerate(index2word)])
    word_embeddings = [[0.0] * EMBEDDING_SIZE, [0.0] * EMBEDDING_SIZE]
    for _, word in enumerate(index2word[2:]):
        word_embeddings.append(model[word].tolist())
    model.save(os.path.join(DATA_DIR, 'word2vec_model.bin'))
    serialize(word2index, os.path.join(DATA_DIR, 'word2index.bin'))
    serialize(word_embeddings, os.path.join(DATA_DIR, 'word_embeddings.bin'))


if __name__ == '__main__':
    main()
    pass
