# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import re
from tensorflow.python.platform import gfile

DATA_DIR = os.path.join('..', 'dataset')
LABEL_DIGIT = {'entailment': 1, 'neutral': 0, 'contradiction': 2}
MAX_LABELS = 5


def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = str.lower(line)
            line = json.loads(line)
            data.append(line)
    return data


def filter_valid_data(data):
    """
    过滤gold label为'-'的数据
    :param data:
    :return:
    """
    return list(filter(lambda x: x['gold_label'] != '-', data))


def create_del_chars():
    del_chars = [chr(c) for c in range(256)]
    del_chars = [x for x in del_chars if not x.isalnum()]
    del_chars.remove(' ')
    del_chars = ''.join(del_chars)
    return del_chars


def parse_data(data):
    new_data = []
    del_chars = create_del_chars()
    for j, d in enumerate(data):
        record = [parse_sentence(d['sentence1'], del_chars),
                  parse_sentence(d['sentence2'], del_chars)]

        gold_label = d['gold_label']
        gold_label = LABEL_DIGIT[gold_label]
        record.append(gold_label)

        annotator_labels = d['annotator_labels']
        annotator_labels = list(filter(lambda x: x != '', annotator_labels))
        annotator_labels = [LABEL_DIGIT[x] for x in annotator_labels]
        for i in range(MAX_LABELS - len(annotator_labels)):
            annotator_labels.append(gold_label)
        record.append(annotator_labels)

        new_data.append(record)

        if j % 1000 == 0:
            print(j)

    return new_data


def parse_sentence(line, del_chars):
    """
    删掉除了空格以外的非数字字母字符
    :param line:
    :param del_chars:
    :return:
    """
    line = line.translate(str.maketrans('', '', del_chars))
    line = re.sub('\s+', ' ', line)  # 将连续的多个空格变为一个
    line = line.split(' ')
    return line


def serialize(data, file_path):
    if gfile.Exists(file_path):
        print('{} already exists'.format(file_path))
        return
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        print('saving {} lines to {}'.format(len(data), file_path))


def deserialize(file_path):
    if not gfile.Exists(file_path):
        raise RuntimeError('{} does not exist'.format(file_path))
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    json_files = ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl']
    bin_files = [j[:-6] + '.bin' for j in json_files]
    for i in range(len(json_files)):
        data_set = read_json_file(os.path.join(DATA_DIR, json_files[i]))
        data_set = filter_valid_data(data_set)
        data_set = parse_data(data_set)
        serialize(data_set, os.path.join(DATA_DIR, bin_files[i]))


if __name__ == '__main__':
    main()
