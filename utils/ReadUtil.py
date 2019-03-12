"""
文件读入工具
"""

import json
import pandas as pd


def get_items(file_path):
    """
    读取文件
     :return:  pandas对象
    """
    data = pd.read_csv(file_path,
                       sep='\t',
                       encoding='gb18030',
                       nrows=None)
    return data


def get_item_names(file_path):
    data = get_items(file_path)
    res = data.loc[:, 'ITEM_NAME']
    return res


def get_item_types(file_path):
    data = get_items(file_path)
    type_data = data.loc[:, 'TYPE']
    return type_data


def read_json2dict(path):
    """
    将json文件读取成字典
    :param path: json文件路径
    :return: 字典
    """
    fp = open(path, 'r')
    return json.loads(fp.readline())


# noinspection PyShadowingBuiltins
def get_classes(classes_file):
    """
    获得类别文件中的类别信息
    :param classes_file: 类别文件路径
    :return: 类别字典
    """
    dic = {}
    with open(classes_file, 'r', encoding="utf-8") as clsFile:
        for line in clsFile:
            t = line.split(' ')
            type_name = ''
            type_index = ''
            ru_gc_len = len(t)
            for i in range(0, ru_gc_len):
                if i < ru_gc_len - 1:
                    type_name += t[i]
                    if i < ru_gc_len - 2:
                        type_name += ' '
                else:
                    type_index = t[i].strip()  # 去除行尾的换行符
            dic[type_name] = type_index
    return dic


import tensorflow


def is_gpu_available(cuda_only=True):
    from tensorflow.python.client import device_lib as _device_lib
    if cuda_only:
        return any((x.device_type == 'GPU')
                   for x in _device_lib.list_local_devices())
    else:
        return any((x.device_type == 'GPU' or x.device_type == 'SYCL')
                   for x in _device_lib.list_local_devices())
