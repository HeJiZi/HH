#!/usr/bin/env python
# encoding: utf-8


import fasttext

if __name__ == '__main__':

    model = fasttext.skipgram('./data/transformed_train.txt', 'model_skipgram')
    print(model.words)  # 打印词向量

    # cbow model
    model = fasttext.cbow('./data/transformed_train.txt', 'model_cbow')
    print(model.words)  # 打印词向量

