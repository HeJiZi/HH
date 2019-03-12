import pandas as pd
import core.preprocess.Cut as cut


class Preprocess:
    _data_file = None
    _encoding = None
    _type_dic = {}
    _word_dic = {}
    _ori_df = None
    _input_df = None

    _maxn = 0
    _minn = 0
    _wordngram = 0
    _type_size = 0
    _word_size = 1
    _level = 3

    def __init__(self, data_file, encoding='gb18030', minn=3, maxn=6, wordngram=1, level=3):
        self._data_file = data_file
        self._encoding = encoding
        self._minn = minn
        self._maxn = maxn
        self._wordngram = wordngram
        self._level = level

    def compile(self):
        """
        开始对文件的编译，将文件转换为二维数组
        :return:二维数组 第一列：类别编号，第二列->最后一列：词向量
        """
        _ori_df = pd.read_csv(self._data_file, sep='\t', encoding=self._encoding, nrows=None)
        ori_df_len = len(_ori_df)
        _inputs_vec = []
        for index, row in _ori_df.iterrows():
            item_name = row[0]
            type = row[1]
            self._add_type(type)
            input_row = [self._type_code(type)]
            words = cut.get_cut().do(item_name, True)
            for word in words:
                self._add_word(word)
                input_row.append(self._word_dic[word])
                input_row.extend(self._compute_sub_word(word))
            input_row.extend(self._compute_ngram_word(words))
            _inputs_vec.append(input_row)
            print('已完成[{0}/{1}]---------------------'.format(index + 1, ori_df_len))
        return _inputs_vec

    def save_word_dic(self, file):
        of = open(file, "w", encoding="utf-8")
        for word in self._word_dic.keys():
            of.write(word.strip() + ' ' + str(self._word_dic[word]) + '\n')
        of.close()

    def save_type_dic(self, file):
        of = open(file, "w", encoding="utf-8")
        for type in self._type_dic.keys():
            of.write(type.strip() + ' ' + str(self._type_dic[type]) + '\n')
        of.close()

    def _add_type(self, type):
        types = type.split('--')
        real_type = "--".join(types[:self._level])
        if real_type not in self._type_dic.keys():
            self._type_dic[real_type] = self._type_size
            self._type_size += 1

    def _type_code(self, type):
        types = type.split('--')
        real_type = "--".join(types[:self._level])
        return self._type_dic[real_type]

    def _add_word(self, word):
        if word not in self._word_dic.keys():
            self._word_dic[word] = self._word_size
            self._word_size += 1
            return self._word_size - 1
        else:
            return self._word_dic[word]

    def _compute_sub_word(self, word):
        """
        计算一个词语的子词语并返回子词语向量
        :param word: 目标词语
        :return: 例如word = "big" min =2 maxn=3 时,返回[1,3], 1:"bi"的id,3:"ig"的id
        """
        w_len = len(word)
        subwords_vec = []
        if w_len <= self._minn:
            return subwords_vec
        for i in range(0, w_len):
            if i + self._minn > w_len:
                break
            last = w_len if (i + self._maxn) > w_len else i + self._maxn
            for j in range(i + self._minn, last + 1):
                subword = word[i:j]
                subwords_vec.append(self._add_word(subword))
        return subwords_vec

    def _compute_ngram_word(self, line):
        """
        计算一行词语的ngramword并返回ngramword向量
        :param line: 一行词语
        :return:例：ngram =2,line=['hi','bye'],返回[4],4:'hibye'的id
        """
        ws_len = len(line)
        if ws_len < self._wordngram or self._wordngram <= 1:
            return []
        result = []
        for i in range(0, ws_len):
            if i + self._wordngram > ws_len:
                break
            ngramword = "".join(line[i:i + self._wordngram])
            result.append(self._add_word(ngramword))
        return result

    @property
    def word_size(self):
        return self._word_size

    @property
    def type_size(self):
        return self._type_size

# def get_vectors(init = True):
#     if(init):
#         preprocess = Preprocess()
#         preprocess.calculate_word_cout()


# Preprocess().init_dict(have_count_file=True)
# print(Preprocess("",minn=3,maxn=4)._compute_sub_word("hello"))
# empty = [0]
# empty.extend([])
# print(empty)

# print(Preprocess("",wordngram=2)._compute_ngram_word(["我","是","一个","大帅哥"]))
# ^\w+$
