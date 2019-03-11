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
    def save_directory(self):
        return os.path.join(self.data_directory, 'save')

    @property
    def record_directory(self):
        return os.path.join(self.data_directory, 'record')

    @property
    def ori_data(self):
        return os.path.join(self.data_directory, 'train.tsv')

    @property
    def stop_words(self):
        return os.path.join(self.data_directory, 'stop_words.txt')

    @property
    def my_dict(self):
        return os.path.join(self.data_directory, 'my_dict.txt')

    @property
    def count(self):
        return os.path.join(self.data_directory, 'count.txt')

    @property
    def dict(self):
        return os.path.join(self.data_directory, 'dict.txt')

    @property
    def custom_dict(self):
        return os.path.join(self.data_directory, 'custom_dict.txt')

    @property
    def record_result(self):
        return self.join(self.record_directory, 'record_result.txt')

    @property
    def type_code(self):
        return self.join(self.save_directory, 'type_code.txt')

    @property
    def word_code(self):
        return self.join(self.save_directory, 'word_code.txt')

    @staticmethod
    def join(path, paths):
        return os.path.join(path, paths)

