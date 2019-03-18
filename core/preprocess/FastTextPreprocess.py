from core.preprocess.Preprocess import Preprocess
import pandas as pd
import core.preprocess.Cut as cut


class FastTextPreprocess(Preprocess):
    _data_table = None
    _cut_off_line = "----------------------------------------------------------------"

    def compile(self):
        _ori_df = pd.read_csv(self._data_file, sep='\t', encoding=self._encoding, nrows=None)
        ori_df_len = len(_ori_df)
        types = []
        item_names = []
        for index, row in _ori_df.iterrows():
            row_words = []
            types.append(row[1])
            item_name = row[0]
            item_names.append(cut.get_cut().do(item_name, True))
            if index & 0xfff == 0:
                print('已完成[{0}/{1}]---------------------'.format(index + 1, ori_df_len))
        print('已完成')
        self._data_table = [item_names, types]
        return self._data_table

    def save(self, file):
        of = open(file, "w", encoding="utf-8")
        for name in self._data_table[0]:
            of.write(' '.join(name) + '\n')
        of.write(self._cut_off_line + '\n')
        of.write("\n".join(self._data_table[1]))
        of.close()

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


from python_example import TreeNode as tn
import python_example as m
import queue


class _TreeNode:

    def __init__(self, name="", node=None):
        if node is None:
            self._tn = tn(name)
        else:
            self._tn = node
        self._children = {}
        self._parent = None

    @property
    def name(self):
        return tn.name(self._tn)

    @property
    def parent(self):
        return self._parent

    def add_chlid(self, node_name):
        if node_name not in self._children.keys():
            node = tn.add_child(self._tn, node_name)
            wrapper_node = _TreeNode(node=node)
            self._children[node_name] = wrapper_node
            wrapper_node._parent = self

    def get_child(self, name):
        return self._children[name]

    def bfs(self):
        m.print_node(self._tn)
        q = queue.Queue()
        for n in self._children.values():
            m.print_node(n._tn)
            q.put(n)
        while not q.empty():
            n = q.get()
            m.print_node(n._tn)
            for c in n._children.values():
                q.put(c)

def bulidTree(types):
    root = create_node("__root__")
    for type in types:
        level_type = type.split('--')
        root.add_chlid(level_type[0])
        p1 = root.get_child(level_type[0])
        level2_name = '--'.join(level_type[:2])
        p1.add_chlid(level2_name)
        p2 = p1.get_child(level2_name)
        p2.add_chlid('--'.join(level_type))
    return root


def create_node(name):
    return _TreeNode(name)


if __name__ == '__main__':
    from utils.PathUtil import Path

    # import fastText
    from sklearn.model_selection import train_test_split

    p = Path()
    f = p.join(p.data_directory, 'tempTrain.txt')
    ftp = FastTextPreprocess(p.ori_data)
    # ftp.compile()
    # ftp.save(f)

    x, y = ftp.load(f)
    a = bulidTree(y)
    a.bfs()


    # m.receiveTree(a)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=520)
    #
    # cls = fastText.fit(x_train, y_train)
    # print(cls.predict_ndarray(x_test, y_test))
