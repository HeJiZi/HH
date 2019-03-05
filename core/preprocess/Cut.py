"""
分词工具
"""
import jieba
import thulac

def get_cut():
    return JiebaCut()


class Cut:
    def do(self,sentence):
        return


class JiebaCut(Cut):
    def __init__(self):    # TODO jieba 分词工具初始化
        return

    def do(self, sentence):
        return jieba.cut(sentence)


class ThuCut(Cut):
    def __init__(self):
        self.thu1 = thulac.thulac(seg_only=True)

    def do(self,sentence):
        return self.thu1.cut(sentence, text=True).split(" ")


