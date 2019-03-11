import core.preprocess.Cut as Cut
import numpy as np
from utils.PathUtil import Path


# 去除停用词后的分词

_stopwords = None
_path = Path()
cut = Cut.get_cut()

def stopwords():
    global _stopwords
    if(_stopwords == None):
        _stopwords = set()
        with open(_path.stop_words, 'r', encoding='utf-8') as wordsFile:
            for line in wordsFile:
                _stopwords.add(line.strip())
    return _stopwords


def  seg_depart(sentence):
    # 对文档中的一行进行中文分词
    sentence_depart = cut.do(sentence.strip())
    # 创建一个停用词列表
    # 输出结果
    departed_str = ''

    # 去停用词
    for word in sentence_depart:
        if not word.isalpha():
            continue
        if word not in stopwords().keys():
            if word != '\t':
                departed_str += word
                departed_str += " "
    return departed_str


def seg_depart_return_generator(sentence, stop_path):
    # 对文档中的每一行进行中文分词
    sentence_depart = cut.do(sentence.strip())
    # 创建一个停用词列表
    # 输出结果
    departed_str = ''
    # 去停用词
    for word in sentence_depart:
        if not word.isalpha():
            continue
        if word not in stopwords().keys():
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


