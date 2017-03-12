# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.contrib as contrib


class MatchLstm:
    def __init__(self, vocab_size, sentence_size, embedding_size,
                 word_embedding, initializer=tf.truncated_normal_initializer(stddev=0.1),
                 session=tf.Session(), num_class=3,
                 window_size=4, name='MatchLstm'):
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._we = word_embedding
        self._initializer = initializer
        self._name = name
        self._num_class = num_class
        self._sess = session
        self._window_size = window_size

        self._build_inputs_and_vars()

        self._inference()

        self._initial_optimizer()

    def _build_inputs_and_vars(self):
        self.premises = tf.placeholder(shape=[None, self._sentence_size], dtype=tf.int32,
                                       name='premises')
        self.hypotheses = tf.placeholder(shape=[None, self._sentence_size], dtype=tf.int32,
                                         name='hypotheses')
        self.labels = tf.placeholder(shape=[None, self._num_class], dtype=tf.float32,
                                     name='labels')
        self.lr = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        self._batch_size = tf.shape(self.premises)[0]

        with tf.variable_scope(self._name):
            self._word_embedding = tf.get_variable(name='word_embedding',
                                                   shape=[self._vocab_size, self._embedding_size],
                                                   initializer=tf.constant_initializer(self._we),
                                                   trainable=False)

        self._embed_pre = self._embed_inputs(self.premises, self._word_embedding)
        self._embed_hyp = self._embed_inputs(self.hypotheses, self._word_embedding)

    def _inference(self):
        with tf.variable_scope('{}_lstm_s'.format(self._name)):
            lstm_s = contrib.rnn.BasicLSTMCell(num_units=self._embedding_size, forget_bias=0.0)
            pre_length = self._length(self.premises)
            h_s, _ = tf.nn.dynamic_rnn(lstm_s, self._embed_pre, sequence_length=pre_length,
                                       dtype=tf.float32)
            self.h_s = h_s

        with tf.variable_scope('{}_lstm_t'.format(self._name)):
            lstm_t = contrib.rnn.BasicLSTMCell(num_units=self._embedding_size, forget_bias=0.0)
            hyp_length = self._length(self.hypotheses)
            h_t, _ = tf.nn.dynamic_rnn(lstm_t, self._embed_hyp, sequence_length=hyp_length,
                                       dtype=tf.float32)
            self.h_t = h_t

        self.lstm_m = contrib.rnn.BasicLSTMCell(num_units=self._embedding_size,
                                                forget_bias=0.0)
        h_m_arr = tf.TensorArray(dtype=tf.float32, size=self._batch_size)

        i = tf.constant(0)
        c = lambda x, y: tf.less(x, self._batch_size)
        b = lambda x, y: self._match_sent(x, y)
        res = tf.while_loop(cond=c, body=b, loop_vars=(i, h_m_arr))

        self.h_m_tensor = tf.squeeze(res[-1].stack(), axis=[1])

        with tf.variable_scope('{}_fully_connect'.format(self._name)):
            w_fc = tf.get_variable(shape=[self._embedding_size, self._num_class],
                                   initializer=self._initializer, name='w_fc')
            b_fc = tf.get_variable(shape=[self._num_class],
                                   initializer=self._initializer, name='b_fc')
            self.logits = tf.matmul(self.h_m_tensor, w_fc) + b_fc

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                                logits=self.logits,
                                                                name='cross_entropy')
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name='cross_entropy_sum')
        self.loss_op = tf.div(cross_entropy_sum, tf.cast(self._batch_size, dtype=tf.float32))

    def _match_sent(self, i, h_m_arr):
        h_s_i = self.h_s[i]
        h_t_i = self.h_t[i]
        length_s_i = self._length(self.premises[i])
        length_t_i = self._length(self.hypotheses[i])

        state = self.lstm_m.zero_state(batch_size=1, dtype=tf.float32)

        k = tf.constant(0)
        c = lambda a, x, y, z, s: tf.less(a, length_t_i)
        b = lambda a, x, y, z, s: self._match_attention(a, x, y, z, s)
        res = tf.while_loop(cond=c, body=b, loop_vars=(k, h_s_i, h_t_i, length_s_i, state))

        final_state_h = res[-1].h
        h_m_arr = h_m_arr.write(i, final_state_h)

        i = tf.add(i, 1)
        return i, h_m_arr

    def _match_attention(self, k, h_s, h_t, length_s, state):
        h_t_k = tf.reshape(h_t[k], [1, -1])
        h_s_j = tf.slice(h_s, begin=[0, 0], size=[length_s, self._embedding_size])

        with tf.variable_scope('{}_attention_w'.format(self._name)):
            w_s = tf.get_variable(shape=[self._embedding_size, self._embedding_size],
                                  initializer=self._initializer, name='w_s')
            w_t = tf.get_variable(shape=[self._embedding_size, self._embedding_size],
                                  initializer=self._initializer, name='w_t')
            w_m = tf.get_variable(shape=[self._embedding_size, self._embedding_size],
                                  initializer=self._initializer, name='w_m')
            w_e = tf.get_variable(shape=[self._embedding_size, 1],
                                  initializer=self._initializer, name='w_e')

        last_m_h = state.h
        sum_h = tf.matmul(h_s_j, w_s) + tf.matmul(h_t_k, w_t) + tf.matmul(last_m_h, w_m)
        e_kj = tf.matmul(tf.tanh(sum_h), w_e)
        a_kj = tf.nn.softmax(e_kj)
        alpha_k = tf.matmul(a_kj, h_s_j, transpose_a=True)
        alpha_k.set_shape([1, self._embedding_size])

        m_k = tf.concat([alpha_k, h_t_k], axis=1)
        with tf.variable_scope('{}_lstm_m'.format(self._name)):
            _, new_state = self.lstm_m(inputs=m_k, state=state)

        k = tf.add(k, 1)
        return k, h_s, h_t, length_s, new_state

    def _embed_inputs(self, inputs, embeddings):
        ndim0_tensor_arr = tf.TensorArray(dtype=tf.float32, size=self._batch_size)
        i = tf.constant(0)
        c = lambda x, y, z, n: tf.less(x, self._batch_size)
        b = lambda x, y, z, n: self._embed_line(x, y, z, n)
        res = tf.while_loop(cond=c, body=b,
                            loop_vars=(i, inputs, embeddings, ndim0_tensor_arr))
        ndim0_tensor = res[-1].stack()
        ndim0_tensor = tf.reshape(ndim0_tensor, [-1, self._sentence_size, self._embedding_size])
        return ndim0_tensor

    def _embed_line(self, i, inputs, embeddings, ndim0_tensor_arr):
        ndim1_list = []
        for j in range(self._sentence_size):
            word = inputs[i][j]
            unk_word = tf.constant(-1)
            f1 = lambda: tf.squeeze(tf.nn.embedding_lookup(params=embeddings, ids=word))
            f2 = lambda: tf.zeros(shape=[self._embedding_size])
            res_tensor = tf.case([(tf.not_equal(word, unk_word), f1)], default=f2)
            ndim1_list.append(res_tensor)
        for j in range(self._sentence_size):
            word = inputs[i][j]
            unk_word = tf.constant(-1)
            f1 = lambda: self._ave_vec(ndim1_list, j)
            f2 = lambda: ndim1_list[j]
            ndim1_list[j] = tf.case([(tf.not_equal(word, unk_word), f2)],
                                    default=f1)
        ndim1_tensor = tf.stack(ndim1_list)
        ndim0_tensor_arr = ndim0_tensor_arr.write(i, ndim1_tensor)
        i = tf.add(i, 1)
        return i, inputs, embeddings, ndim0_tensor_arr

    def _ave_vec(self, embed_list, cur_pos):
        """
        生词的词向量为词窗口的词向量平均值
        :param embed_list:
        :param cur_pos:
        :return:
        """
        left_pos = max(0, cur_pos - self._window_size)
        right_pos = min(cur_pos + self._window_size, self._sentence_size)
        e_list = embed_list[left_pos:cur_pos] + embed_list[cur_pos + 1:right_pos + 1]
        e_tensor = tf.stack(e_list)
        ave_tensor = tf.reduce_mean(e_tensor, axis=0)
        return ave_tensor

    @staticmethod
    def _length(sequence):
        mask = tf.sign(tf.abs(sequence))
        length = tf.reduce_sum(mask, axis=-1)
        return length

    def _initial_optimizer(self):
        with tf.variable_scope('{}_step'.format(self._name)):
            self.global_step = tf.get_variable(shape=[],
                                               initializer=tf.constant_initializer(0),
                                               dtype=tf.int32,
                                               name='global_step')
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
        self.train_op = self._optimizer.minimize(self.loss_op, global_step=self.global_step)


if __name__ == '__main__':
    with tf.Session() as sess:
        embedding = np.random.randn(4, 6)
        embedding[0] = 0.0
        model = MatchLstm(vocab_size=7, sentence_size=5, embedding_size=6,
                          word_embedding=embedding, session=sess)
        model.batch_size = 1
        sent1 = [[3, -1, 2, 1, 0],
                 [4, 5, 1, 0, 0],
                 [2, 1, 0, 0, 0]]

        sent2 = [[2, 1, 0, 0, 0],
                 [3, -1, 2, 1, 0],
                 [4, 5, 1, 0, 0]]

        labels = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]

        sess.run(tf.global_variables_initializer())
        for temp in range(300):
            loss, _, step = sess.run([model.loss_op, model.train_op, model.global_step],
                                     feed_dict={model.premises: sent1, model.hypotheses: sent2,
                                                model.labels: labels, model.lr: 0.001})
            print(step, loss)
            sent1, sent2 = sent2, sent1
