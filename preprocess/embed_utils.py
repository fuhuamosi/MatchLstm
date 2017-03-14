# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from preprocess.file_utils import serialize, deserialize, DATA_DIR
import os
from gensim.models import Word2Vec

from glove import Glove
from glove import Corpus

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


def embed_word2vec():
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


def embed_glove2():
    # prepare_corpus()
    corpus = read_corpus(CORPUS_PATH)
    corpus_model = Corpus()
    corpus_model.fit(corpus, window=5)
    glove = Glove(no_components=300)
    glove.fit(corpus_model.matrix, epochs=int(20),
              no_threads=8, verbose=True)
    dic = corpus_model.dictionary
    index2word = [_PAD, _NULL] + sorted(dic, key=dic.get, reverse=False)
    word2index = dict([(y, x) for (x, y) in enumerate(index2word)])
    word_embeddings = [[0.0] * EMBEDDING_SIZE, [0.0] * EMBEDDING_SIZE]
    for i in range(2, len(word2index)):
        index = i - 2
        word_embeddings.append(glove.word_vectors[index].tolist())
    serialize(word2index, os.path.join(DATA_DIR, 'word2index_glove.bin'))
    serialize(word_embeddings, os.path.join(DATA_DIR, 'word_embeddings_glove.bin'))


def embed_glove(path):
    model = {}
    with open(path, 'r') as f:
        for line in f:
            glove_list = line.strip().split()
            word = glove_list[0]
            vec = [float(x) for x in glove_list[1:]]
            model[word] = vec
    index2word = [_PAD, _NULL] + list(model.keys())
    word2index = dict([(y, x) for (x, y) in enumerate(index2word)])
    word_embeddings = [[0.0] * EMBEDDING_SIZE, [0.0] * EMBEDDING_SIZE]
    for _, word in enumerate(index2word[2:]):
        word_embeddings.append(model[word].tolist())
    serialize(word2index, os.path.join(DATA_DIR, 'word2index_glove.bin'))
    serialize(word_embeddings, os.path.join(DATA_DIR, 'word_embeddings_glove.bin'))


if __name__ == '__main__':
    # embed_word2vec()
    embed_glove(os.path.join(DATA_DIR, ''))
    pass
