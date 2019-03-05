import jieba
import numpy as np


# 去除停用词后的分词
from utils.PathUtil import Path


def seg_depart(sentence, stop_path):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = [line.strip() for line in open(stop_path, encoding='utf-8').readlines()]
    # 输出结果
    departed_str = ''
    # 去停用词
    for word in sentence_depart:
        if not word.isalpha():
            continue
        if word not in stopwords:
            if word != '\t':
                departed_str += word
                departed_str += " "
    return departed_str


def seg_depart_return_generator(sentence, stop_path):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = [line.strip() for line in open(stop_path, encoding='utf-8').readlines()]
    # 输出结果
    departed_str = ''
    # 去停用词
    for word in sentence_depart:
        if not word.isalpha():
            continue
        if word not in stopwords:
            if word != '\t':
                departed_str += word
                departed_str += " "
    return (i for i in departed_str.split(" "))


def cut_to_array(items):
    toks = []
    for item in items:
        toks.append(seg_depart_return_generator(item, Path().stop_words))
    toks = np.array(toks)

    return toks
