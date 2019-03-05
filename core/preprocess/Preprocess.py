import classData as  c
# print(c.getClasses("./train.tsv"))
import jieba
import jieba.analyse
import pandas as pd
import random

trainPath = './data/train.tsv'
idfPath = "./data/idf.txt.big"
vecsPath = "./data/vectors.csv"

# segPath = './data/segment'
# modelPath = './model/baike.model'
# vecPath = './model/word2vec_format'
encode = 'utf-8'

def turnFeaVecs(items, wordDic):
    maxLength = 0
    vecs=[]
    clsDic = c.getClasses(trainPath)

    # 对x进行预处理
    for index in range(len(items)):
        tags = jieba.analyse.extract_tags(items.loc[index,0])
        tagsLen = len(tags)
        if tagsLen>maxLength:
            maxLength = tagsLen
        vec = []
        for tag in tags:
            vec.append(wordDic[tag] / len(wordDic)) # 对数据进行归一化
        vecs.append(vec)

    # 对数据顺序进行预处理
    for index in range(len(vecs)):
        xlen = len(vecs[i])
        dlen = maxLength - xlen  # 长度不够填充空值
        while dlen > 0:
            vecs[index].append(None)
            dlen = -1
        # 打乱词的顺序
        random.shuffle(vecs[i])
        # 添加y
        cls = items.loc[index, 1].split('--', 1)
        vecs[i].insert(0, clsDic[cls])

    return vecs