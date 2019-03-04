from utils.DataAccess import DataAccess
import jieba.analyse
import math


# 创建停用词列表
def get_stopwords_list():
    return [line.strip() for line in open('./data/stop_words.txt', encoding='utf-8').readlines()]


# 去除停用词后的分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())   # strip  移除字符串头尾指定的字符序列。默认为空格
    # 创建一个停用词列表
    stopwords = get_stopwords_list()
    # 输出结果
    departed_str = ''
    # 去停用词
    for word in sentence_depart:
        if not word.isalpha():    # 去除数字
            continue
        if word not in stopwords:
            if word != '\t':
                departed_str += word
                departed_str += " "

    return departed_str


# 返回一个元组列表,强制转化成字典
def sort_dict(d):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))


# 字典去重
def dict_delete(word, weight):
    res = {}
    print("一共" + str(len(word)) + "个特征")
    for j in range(len(word)):
        if j % 256 == 0:
            print("正在处理第" + str(j) + "个特征")
        res[word] = weight[j]

    return res


# 计算词频
def my_tf(tag, item):

    tags = item.split()
    count = 0
    for i in tags:
        if i == tag:
            count += 1
    return count / len(tags)


def my_df(corpus):
    IDFdict = {}
    countDict = {}
    for item in corpus:
        for tag in item.split():
            if tag in countDict.keys():
                countDict[tag] += 1
            else:
                countDict[tag] = 1

    # for tag in countDict.keys():
    #     IDFdict[tag] = countDict[tag] / LEN
    return countDict
    # return IDFdict


if __name__ == "__main__":

    corpus = []
    data = DataAccess.get_item_names("./data/train.tsv")
    LEN = len(data)
    index = 0
    for item in data:
        index += 1
        if index % 1024 == 0:
            print("正在切分第"+str(index)+"数据\n")
        item = seg_depart(item)
        # tags = jieba.analyse.extract_tags(item, topK=5)
        text = " ".join(str(i) for i in item.split(" "))
        corpus.append(text)

    print("开始计算DF\n")

    index = 0
    IDFdict = my_df(corpus)
    IDFdict = sort_dict(IDFdict)

    fp = open('./count.txt', 'w')
    for j in IDFdict:
        fp.write(j+' '+str(IDFdict[j])+'\n')
    fp.close()
    print("完成")

