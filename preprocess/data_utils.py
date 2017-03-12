# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict
from preprocess.embed_utils import PAD_ID, NULL_ID
from preprocess.file_utils import deserialize, DATA_DIR


def vectorize_data(data, word2idx: Dict, max_sent_size):
    new_data = []
    for d in data:
        sent1 = d[0]
        sent1 = [word2idx.get(s, -1) for s in sent1]
        sent1.append(NULL_ID)  # premise需要在结尾加一个null word
        pad_length = max_sent_size - len(sent1)
        sent1.extend([PAD_ID] * pad_length)

        sent2 = d[1]
        sent2 = [word2idx.get(s, -1) for s in sent2]
        pad_length = max_sent_size - len(sent2)
        sent2.extend([PAD_ID] * pad_length)

        line = [sent1, sent2, d[2], d[3]]
        new_data.append(line)
    return new_data


if __name__ == '__main__':
    # valid_data = deserialize('../dataset/snli_1.0_dev.bin')
    # word_inx = deserialize('../dataset/word2index.bin')
    # valid_data = vectorize_data(valid_data, word_inx, 80)
    pass
