# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from preprocess.data_utils import vectorize_data
from preprocess.file_utils import deserialize
from app.decorator import exe_time
from model.match_lstm import MatchLstm

import tensorflow as tf
import os
from sklearn import metrics
import numpy as np

tf.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
tf.flags.DEFINE_float('decay_ratio', 0.95, 'Learning rate decay ratio')
tf.flags.DEFINE_float('min_lr', 1e-6, 'Min learning rate')
tf.flags.DEFINE_float('max_grad_norm', 40.0, 'Clip gradients to this norm.')
tf.flags.DEFINE_integer('evaluation_interval', 100, 'Evaluate and print results every x epochs')
tf.flags.DEFINE_integer('batch_size', 30, 'Batch size for training.')
tf.flags.DEFINE_integer('sent_size', 80, 'Max sentence size.')
tf.flags.DEFINE_integer('num_class', 3, 'Max sentence size.')
tf.flags.DEFINE_integer('epochs', 20000, 'Number of epochs to train for.')
tf.flags.DEFINE_integer('embedding_size', 300, 'Embedding size for embedding matrices.')
tf.flags.DEFINE_string('data_dir', os.path.join('..', 'dataset'), 'Directory containing dataset')

FLAGS = tf.flags.FLAGS


@exe_time
def load_data():
    train_data = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_train.bin'))
    valid_data = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_dev.bin'))
    test_data = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_test.bin'))
    return train_data, valid_data, test_data


@exe_time
def load_dict():
    word2index = deserialize(os.path.join(FLAGS.data_dir, 'word2index.bin'))
    word_embeddings = deserialize(os.path.join(FLAGS.data_dir, 'word_embeddings.bin'))
    return word2index, word_embeddings


@exe_time
def vectorize_all_data(train_data, valid_data, test_data, word2index):
    train_data = vectorize_data(train_data, word2index, max_sent_size=FLAGS.sent_size)
    valid_data = vectorize_data(valid_data, word2index, max_sent_size=FLAGS.sent_size)
    test_data = vectorize_data(test_data, word2index, max_sent_size=FLAGS.sent_size)
    return train_data, valid_data, test_data


def cnt2ratio(label_list):
    ratio_list = [0.0] * FLAGS.num_class
    for label in label_list:
        ratio_list[label] += 1
    ratio_list = [x / 5 for x in ratio_list]
    return ratio_list


def unpack_data(data):
    new_data = list(zip(*data))

    labels = list(new_data[-1])
    labels = [cnt2ratio(l) for l in labels]
    new_data[-1] = labels

    return new_data


def random_choice(data, size):
    data_len = len(data)
    sub = np.random.choice(range(data_len), size=size)
    sample_data = [data[s] for s in sub]
    return sample_data


def run_epoch(sess, model, train_data, train_len, loss_list, i):
    start = i * FLAGS.batch_size % train_len
    end = (i + 1) * FLAGS.batch_size % train_len
    if start < end:
        data = train_data[start:end]
    else:
        data = train_data[start:] + train_data[:end]
    data = unpack_data(data)
    loss, _ = sess.run([model.loss_op, model.train_op],
                       feed_dict={model.premises: data[0],
                                  model.hypotheses: data[1],
                                  model.labels: data[3],
                                  model.lr: FLAGS.learning_rate})
    loss_list.append(loss)


@exe_time
def validate_data(sess, data, model, name):
    data = unpack_data(data)
    preds = []
    for i in range(len(data[0])):
        pred = sess.run(model.predict_op,
                        feed_dict={model.premises: data[0][i:i + 1],
                                   model.hypotheses: data[1][i:i + 1]})
        preds.extend(pred)
    precision = metrics.accuracy_score(y_true=data[2], y_pred=preds)
    print('The precision of data {} is {}'.format(name, precision))


def main(_):
    train_data, valid_data, test_data = load_data()
    word2index, word_embeddings = load_dict()
    train_data, valid_data, test_data = vectorize_all_data(train_data, valid_data, test_data,
                                                           word2index)
    lr = FLAGS.learning_rate
    with tf.Session() as sess:
        model = MatchLstm(vocab_size=len(word2index), sentence_size=FLAGS.sent_size,
                          embedding_size=FLAGS.embedding_size,
                          word_embedding=word_embeddings, session=sess,
                          initial_lr=lr)
        sess.run(tf.global_variables_initializer())

        train_len = len(train_data)
        loss_list = []
        for i in range(FLAGS.epochs):
            run_epoch(sess, model, train_data, train_len, loss_list, i)

            if i % FLAGS.evaluation_interval == 0:
                loss_arr = np.array(loss_list)
                print('Epoch {} loss mean, std: {}  {}'.format(i, np.mean(loss_arr),
                                                               np.std(loss_arr)))
                loss_list.clear()

                sample_size = 1000
                sample_valid = random_choice(valid_data, size=sample_size)
                sample_test = random_choice(test_data, size=sample_size)

                print('\n')
                validate_data(sess, sample_valid, model, name='valid')
                validate_data(sess, sample_test, model, name='test')

                if lr > FLAGS.min_lr:
                    lr *= FLAGS.decay_ratio
                    sess.run(model.lr_update_op, feed_dict={model.new_lr: lr})
                    print('New learning rate is {}'.format(lr))

                print('\n')


if __name__ == '__main__':
    tf.app.run()
