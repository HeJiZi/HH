from core.preprocess.Preprocess import Preprocess
import core.preprocess.Cut as cut
import pandas as pd
import numpy as np


class PreprocessPlus(Preprocess):
    def compile(self):
        _ori_df = pd.read_csv(self._data_file, sep='\t', encoding=self._encoding, nrows=None)
        ori_df_len = len(_ori_df)
        inputs_vec = []
        input_words = []
        for index, row in _ori_df.iterrows():
            item_name = row[0]
            type = row[1]
            self._add_type(type)
            line = []
            words = cut.get_cut().do(item_name, True)
            for word in words:
                self._add_word(word)
                line.append(word)
            inputs_vec.append([self._type_code(type)])
            input_words.append(line)
            if index & 0xfff == 0:
                print('已分词[{0}/{1}]---------------------'.format(index + 1, ori_df_len))
        for i in range(0,ori_df_len):
            input_row =[]
            for word in input_words[i]:
                input_row.append(self._word_dic[word])
                input_row.extend(self._compute_sub_word(word))
            inputs_vec[i].extend(input_row)
        self._input_df = pd.DataFrame(inputs_vec).fillna(0).astype(np.int32)
        print('已完成')
        return self._input_df
