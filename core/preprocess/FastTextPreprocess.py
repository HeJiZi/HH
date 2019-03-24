import pandas as pd
import core.preprocess.Cut as cut


class FastTextPreprocess:
    _cut_off_line = "----------------------------------------------------------------"

    def __init__(self, data_file, encoding='gb18030'):
        self._data_table = None
        self._ori_np = pd.read_csv(data_file, sep='\t', encoding=encoding, nrows=None).values

    @staticmethod
    def _type_name(name, level):
        types = name.split('--')
        return "--".join(types[:level])

    def compile(self, level=3):
        ori_df_len = len(self._ori_np)
        types = []
        item_names = []
        for index in range(0, ori_df_len):
            types.append(self._type_name(self._ori_np[index, 1], level))
            item_name = self._ori_np[index, 0]
            item_names.append(cut.get_cut().do(item_name, True))
            if index & 0xfff == 0:
                print('分词中[{0}/{1}]---------------------'.format(index + 1, ori_df_len))
        print('分词已完成')
        self._data_table = [item_names, types]
        return self._data_table

    def save(self, file):
        of = open(file, "w", encoding="utf-8")
        for name in self._data_table[0]:
            of.write(' '.join(name) + '\n')
        of.write(self._cut_off_line + '\n')
        of.write("\n".join(self._data_table[1]))
        of.close()

    def update_type(self, level):
        ori_df_len = len(self._ori_np)
        for index in range(0, ori_df_len):
            self._data_table[1][index] = self._type_name(self._ori_np[index, 1], level)
            if index & 0xfff == 0:
                print('更新类别中[{0}/{1}]---------------------'.format(index + 1, ori_df_len))
        print('更新已完成')

    def load(self, file):
        item_names = []
        types = []
        isName = True
        with open(file, 'r', encoding="utf-8") as rf:
            for line in rf:
                if line.strip() == self._cut_off_line:
                    isName = False
                    continue
                if isName:
                    item_names.append(line.strip().split())
                else:
                    types.append(line.strip())
        self._data_table = [item_names, types]
        return self._data_table


# def bulid_tree(types):
#     import fastText
#     root = fastText.create_root_node()
#     for type in types:
#         level_type = type.split('--')
#         root.add_chlid(level_type[0])
#         p1 = root.get_child(level_type[0])
#         level2_name = '--'.join(level_type[:2])
#         p1.add_chlid(level2_name)
#         p2 = p1.get_child(level2_name)
#         p2.add_chlid('--'.join(level_type))
#     return root


if __name__ == '__main__':
    from utils.PathUtil import Path
    import time

    import fastText
    from sklearn.model_selection import train_test_split

    p = Path()
    f = p.join(p.data_directory, 'tempTrain.txt')
    ftp = FastTextPreprocess(p.train80_1, encoding="utf-8")
    model = p.join(p.model_directory, 'temp_model.fs')

    # ftp.compile()
    # ftp.save(f)

    x, y = ftp.load(f)
    # ftp.update_type(level=1)

    # a = bulidTree(y)
    # a.bfs()

    #
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=520)
    #
    # cls = fastText.fit(x_train, y_train, wordNgrams=1, epoch=7)
    # cls.save_model(model)
    cls = fastText.load_model(model)

    vs = cls.predict_prob(x_test)

    # print(cls.predict_ndarray(x_test, y_test))
