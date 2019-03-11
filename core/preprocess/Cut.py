"""
分词工具
"""
import jieba
import thulac

from utils.PathUtil import Path

_cut = None


def get_cut():
    global _cut
    if _cut is None:
        _cut = JiebaCut()
    return _cut


class Cut:
    _stop_words = None
    _path = Path()

    @property
    def stop_words(self):
        if self._stop_words is None:
            self._stop_words = set()
            with open(self._path.stop_words, 'r', encoding='utf-8') as wordsFile:
                for line in wordsFile:
                    self._stop_words.add(line.strip())
        return self._stop_words

    def do(self, sentence, stopwords=False):
        return



class JiebaCut(Cut):
    def __init__(self):  # TODO jieba 分词工具初始化
        jieba.load_userdict(Path().my_dict)
        return

    def do(self, sentence, stopwords=False):
        if not stopwords:
            return jieba.cut(sentence)
        else:
            words = self.do(sentence)
            result = []
            for word in words:
                if word.isalpha() and word not in self.stop_words:
                    result.append(word)
            return result


class ThuCut(Cut):
    def __init__(self):
        self.thu1 = thulac.thulac(seg_only=True)

    def do(self, sentence, stopwords=False):
        return self.thu1.cut(sentence, text=True).split(" ")


# print(JiebaCut().do("腾讯QQ黄钻三个月QQ黄钻3个月季卡官方自动充值可查时间可续费",stopwords= True))
