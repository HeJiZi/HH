"""
时间工具 提供用来计算运行时间的方法
"""

import time


def count_time():
    """
    将该函数赋值给一个变量，在要计算运算时间的代码前后分别添加next(变量名)来显示运算时间。
    """
    a = time.time()
    yield a
    b = time.time()
    print(b - a)
    yield b


def print_time():
    print(time.strftime('%Y.%m.%d.%H.%I.%M.%S', time.localtime(time.time())))


def get_formate_time():
    return time.strftime('%Y-%m-%d_%H.%I.%M', time.localtime(time.time()))

