import random

from utils.PathUtil import Path
import time
import fastText
from sklearn.model_selection import train_test_split
from core.preprocess.FastTextPreprocess import FastTextPreprocess
import utils.TimeUtil as tu
import core.preprocess.Cut as cut


class ShrinkPreoprocess(FastTextPreprocess):

    shrink_set = {
                  '汽配用品'
                  }

    def __init__(self, data_file, encoding='gb18030'):
        super().__init__(data_file, encoding=encoding)
        self.type_divid= None

    def set_divid(self,sets):
        self.type_divid = sets

    def _type_name(self, name, level):
        # types = name.split('--')
        # if types[0].strip() not in ShrinkPreoprocess.shrink_set:
        #     return 'shrink_class'
        # return "--".join(types[:level])
        index = 0
        for d in self.type_divid:
            if name in d:
                return '类别-'+str(index)
            index += 1
        print(name)
        return 'unKnown'

    def compile(self, level=3):
        ori_df_len = len(self._ori_np)
        types = []
        item_names = []
        for index in range(0, ori_df_len):
            t_name = self._type_name(self._ori_np[index, 1], level)
            # if t_name == 'shrink_class':
            #     continue
            types.append(t_name)
            item_name = self._ori_np[index, 0]
            item_names.append(cut.get_cut().do(item_name, True))
            if index & 0xfff == 0:
                print('分词中[{0}/{1}]---------------------'.format(index + 1, ori_df_len))
        print('分词已完成')
        self._data_table = [item_names, types]
        return self._data_table


class Prediction:
    def __init__(self, name, true_type, predict_type, probs):
        self._name = " ".join(name)
        self._true_type = true_type
        self._predict = predict_type
        self._probes = ""
        for i in range(0, len(probs)):
            self._probes += probs[i][0]+":" + str(probs[i][1]) + " "

    def to_csv_line(self):
        return ",".join([self._name, self._true_type, self._predict, self._probes])+"\n"

    @property
    def true_type(self):
        return self._true_type


def create_label_dict(labels):
    dict = {}
    for i in range(0, len(labels)):
        dict[labels[i]] = i
    return dict


def predict(cls, x_test, y_test, out_file):
    labels = cls.get_labels()
    l_dict = create_label_dict(labels)
    probs = cls.predict_prob(x_test)
    rightNum = 0

    of = open(out_file, "w", encoding="utf-8")
    of.write(','.join(["商品名", "真实类别", "预测类别", "预测概率分布(最大四名)"])+'\n')

    preds =[]
    for i in range(0, len(probs)):
        max = probs[i][0]
        maxIndex = 0
        for j in range(1, len(labels)):
            if probs[i][j] > max:
                max = probs[i][j]
                maxIndex = j
        if maxIndex != l_dict[y_test[i]]:
            dict = {}
            for k in range(0, len(labels)):
                dict[labels[k]] = round(probs[i][k], 2)
            prb_disp = sorted(dict.items(), key=lambda d: d[1], reverse=True)[:4]
            preds.append(Prediction(x_test[i], y_test[i], labels[maxIndex], prb_disp))
        else:
            rightNum += 1
    preds = sorted(preds, key=lambda pred: l_dict[pred.true_type], reverse=True)
    for pred in preds:
        of.write(pred.to_csv_line())
    of.close()
    print("正确数：", rightNum, "总数：", len(probs))


def get_type_divid():
    type_divide_file = p.join(p.save_directory, 'type_divid.txt')
    of = open(type_divide_file, "r")
    result = []
    for i in range(0, 3):
        result.append(set(of.readline().strip().split(',')))
    of.close()
    return result

if __name__ == '__main__':
    p = Path()
    f = p.join(p.data_directory, 'tempTrain.txt')
    train_save = p.join(p.save_directory, 'train_801.txt')
    test_save = p.join(p.save_directory, 'test_201.txt')
    model = p.join(p.model_directory, 'temp_model.fs')
    wrongExamples = p.join(p.analyse_directory, 'wrongExamples'+tu.get_formate_time()+'.csv')

    trainP = FastTextPreprocess(p.train80_1, encoding="utf-8")
    testP = FastTextPreprocess(p.test20_1, encoding="utf-8")

    # trainP.set_divid(get_type_divid())
    # testP.set_divid(get_type_divid())
    # x_train, y_train = trainP.compile(level=3)
    # x_test, y_test = testP.compile(level=3)
    # trainP.save(train_save)
    # testP.save(test_save)


    x_train, y_train = trainP.load(train_save)
    x_test, y_test = testP.load(test_save)

    trainP.update_type(level=3)
    testP.update_type(level=3)
    #
    # trainP.save(train_save)
    # testP.save(test_save)

    cc = list(zip(x_train, y_train))
    random.shuffle(cc)
    x_train[:], y_train[:] = zip(*cc)

    a = [
        ['童书', 0.2],
        ['中小学教辅', 0.25],
        ['考试', 0.4],
        ['建筑', 0.5],
        ['文学', 0.6],
        ['小说', 0.6],
        ['大中专教材教辅', 0.6],
        ['历史', 0.6],
        ['艺术', 0.6],
        ['文化用品', 0.7],
        ['进口原版', 0.7],
        ['工业技术', 0.8],
        # ['旅游/地图', 2],
        # ['哲学/宗教', 2],
        # ['健身与保健', 2],
        # ['青春文学', 2],
        # ['经济', 2],
        # ['金融与投资', 2],
        # ['传记', 2],
        # ['文化', 2],
        # ['育儿/家教', 2],
    ]

    cls = fastText.fit(x_train, y_train, wordNgrams=2, epoch=7, sampleWeight=[])
    cls.save_model(model)
    # cls = fastText.load_model(model)
    # print(cls.predict_ndarray(x_test,y_test))
    predict(cls, x_test=x_test, y_test=y_test, out_file=wrongExamples)

    # labels = cls.get_labels()
    # type_divide_file = p.join(p.save_directory, 'type_divid.txt')
    # of = open(type_divide_file, "w")
    # of.write(",".join(labels[:12])+'\n')
    # of.write(",".join(labels[12:78])+'\n')
    # of.write(",".join(labels[78:])+'\n')


