from core.preprocess.FastTextPreprocess import FastTextPreprocess


class TensorFlowPreprocess(FastTextPreprocess):
    def __init__(self, data_file, encoding='gb18030'):
        super().__init__(data_file, encoding=encoding)
        self._word_dict = {}
        self._label_dict = {}

    @staticmethod
    def fit_dict(content, to_dict, isSingle = False):
        key_num = 0
        for name in content:
            index = 0
            if isSingle is False:
                for word in name:
                    if word not in to_dict.keys():
                        to_dict[word] = key_num
                        key_num += 1
            else:
                if name not in to_dict.keys():
                    to_dict[name] = key_num
                    key_num += 1

    def compile(self, level=3):
        super().compile(level=level)
        self.fit_dict(self._data_table[0], self._word_dict)
        self.fit_dict(self._data_table[1], self._label_dict, isSingle=True)
        return self._data_table

    @property
    def word_dict(self):
        return self._word_dict

    @property
    def label_dict(self):
        return self._label_dict



