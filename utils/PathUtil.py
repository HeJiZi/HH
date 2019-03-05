"""
路径工具，获取路径
"""
import os

seg = "\\"


class Path:
    root = ''

    def __init__(self):
        cur_path = os.path.abspath(os.path.dirname(__file__))
        self.root = cur_path[:cur_path.find("HH\\") + len("HH")].replace("\\", seg)

    @property
    def data_directory(self):
        return os.path.join(self.root, 'data')

    @property
    def ori_data(self):
        return os.path.join(self.data_directory, 'train.tsv')

    @property
    def stop_words(self):
        return os.path.join(self.data_directory, 'stop_words.txt')

    @staticmethod
    def join(path, paths):
        return os.path.join(path, paths)
