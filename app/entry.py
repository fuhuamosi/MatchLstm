# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from preprocess.data_utils import vectorize_data
from preprocess.file_utils import deserialize
from app.decorator import exe_time
from model.match_lstm import MatchLstm

import tensorflow as tf
import os
import numpy as np

tf.flags.DEFINE_float('learning_rate', 0.001, '')
tf.flags.DEFINE_float('decay_ratio', 0.95, '')
tf.flags.DEFINE_float('max_grad_norm', 40.0, 'Clip gradients to this norm.')
tf.flags.DEFINE_integer('evaluation_interval', 10, 'Evaluate and print results every x epochs')
tf.flags.DEFINE_integer('batch_size', 30, 'Batch size for training.')
tf.flags.DEFINE_integer('sent_size', 80, 'Max sentence size.')
tf.flags.DEFINE_integer('num_class', 3, 'Max sentence size.')
tf.flags.DEFINE_integer('embedding_size', 300, 'Embedding size for embedding matrices.')
tf.flags.DEFINE_string('data_dir', os.path.join('..', 'dataset'), 'Directory containing dataset')

FLAGS = tf.flags.FLAGS


@exe_time
def load_data():
    train_data = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_train.bin'))
    valid_data = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_dev.bin'))
    test_data = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_test.bin'))
    print('Loading data finished.')
    return train_data, valid_data, test_data


@exe_time
def load_dict():
    word2index = deserialize(os.path.join(FLAGS.data_dir, 'word2index.bin'))
    word_embeddings = deserialize(os.path.join(FLAGS.data_dir, 'word_embeddings.bin'))
    print('Loading dict finished.')
    return word2index, word_embeddings


@exe_time
def vectorize_all_data(train_data, valid_data, test_data, word2index):
    train_data = vectorize_data(train_data, word2index, max_sent_size=FLAGS.sent_size)
    valid_data = vectorize_data(valid_data, word2index, max_sent_size=FLAGS.sent_size)
    test_data = vectorize_data(test_data, word2index, max_sent_size=FLAGS.sent_size)
    print('Vectorize data finished.')
    return train_data, valid_data, test_data


def cnt2ratio(label_list):
    ratio_list = [0.0] * FLAGS.num_class
    for label in label_list:
        ratio_list[label] += 1
    ratio_list = [x / 5 for x in ratio_list]
    return ratio_list


@exe_time
def unpack_data(data):
    new_data = list(zip(*data))

    labels = list(new_data[-1])
    labels = [cnt2ratio(l) for l in labels]
    new_data[-1] = labels

    return new_data


@exe_time
def run_epoch(sess, data, model):
    loss, _, step = sess.run([model.loss_op, model.train_op, model.global_step],
                             feed_dict={model.premises: data[0],
                                        model.hypotheses: data[1],
                                        model.labels: data[3],
                                        model.lr: FLAGS.learning_rate})
    print(loss)


def main(_):
    train_data, valid_data, test_data = load_data()
    word2index, word_embeddings = load_dict()
    train_data, valid_data, test_data = vectorize_all_data(train_data, valid_data, test_data,
                                                           word2index)
    temp_data = unpack_data(valid_data[:30])

    with tf.Session() as sess:
        model = MatchLstm(vocab_size=len(word2index), sentence_size=FLAGS.sent_size,
                          embedding_size=FLAGS.embedding_size,
                          word_embedding=word_embeddings, session=sess)
        sess.run(tf.global_variables_initializer())
        run_epoch(sess, temp_data, model)


if __name__ == '__main__':
    tf.app.run()
