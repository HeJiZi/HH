import jieba
import pandas as pd
import numpy as np


class jiebaExtend:

    # 去除停用词后的分词
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def cut_to_array(items):
        toks = []
        for item in items:
            toks.append(jiebaCut.seg_depart_return_generator(item, './data/stop_words.txt'))
        toks = np.array(toks)

        return toks
