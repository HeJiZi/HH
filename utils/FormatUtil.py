"""
格式转换工具
"""
import pandas as pd
from sklearn.utils import shuffle
import re
import utils.DrawUtil as du
import utils.ReadUtil as ru
import utils.CutExtend as CutExtend
from utils.PathUtil import Path


def filter_out_classes(file, level=1, method=0):
    """
    输入原始数据,为所有类别进行编号，并返回相应的字典
    :param file:原始数据文件名
    :param level:类别级数
    :param method: 为0时转换为[本地生活-- 0] 为1时转换为[本地生活 0]
    :return:    key:类别名   res_dict[key]:类别编号
    """

    code = 'gb18030'
    end = None

    data = pd.read_csv(file,
                       sep='\t',
                       encoding=code,
                       nrows=end
                       )

    # 只获取类别列
    type_data = data.loc[:, 'TYPE']

    res_dict = {}
    index = 0

    for i in range(len(type_data)):
        classes = type_data[i].split("--")
        cls_name = ''

        # 取出对应级别的类别名
        for j in range(level):
            cls_name += classes[j]
            # j+1 == currentLevel
            if j + 1 < len(classes) and method == 0:
                cls_name += '--'

        # 放入字典
        if not cls_name in res_dict.keys():
            res_dict[cls_name.strip()] = index
            index += 1

    return res_dict


def transfer_to_ft_format(file_path, output_path, class_path, method=0, train_file_name="train.txt",
                          test_file_name="test.txt"):
    """
    将数据转换为fastText的指定格式
    :param method:转换类别编码的算法
    :param test_file_name: mean as name
    :param train_file_name: mean as name
    :param file_path: 数据文件的路径
    :param output_path: 导出文件的目录路径
    :param class_path: 类别文件的路径
    """
    code = 'gb18030'
    end = None
    dic = ru.get_classes(class_path)
    data = pd.read_csv(file_path,
                       sep='\t',
                       encoding=code,
                       nrows=end
                       )

    df = shuffle(data)  # 打乱顺序

    du.print_sep()
    print('正在转换类别编码....')
    count = 0

    if method == 1:
        level_dics = []
        for index, row in df.iterrows():
            type = row['TYPE']
            row["TYPE"] = '_label_' + dic[type]
            count += 1
            if count % 10000 == 0:
                print('已转化:%.2f%%' % ((count / len(df)) * 100))
    else:
        keys = dic.keys()
        klen = len(keys)
        for key in keys:
            df.replace(regex=re.compile('^' + key + '[\s\S]*'), value='_label_' + str(dic[key]), inplace=True)
            count += 1
            print('已转化:%.2f%%' % ((count / klen) * 100))

    du.print_sep()
    print('正在分词....')
    for index, row in df.iterrows():
        name = row['ITEM_NAME']
        row['ITEM_NAME'] = CutExtend.seg_depart(name)
        # row['ITEM_NAME'] = " ".join(jieba.cut(name, cut_all=True))
    print('分词完成')
    du.print_sep()

    # 重新对打乱后的dataFrame 进行编号
    df.index = range(len(df))
    traindf = df.loc[:400000, :]
    testdf = df.loc[400001:500000, :]

    traindf.to_csv(Path().join(output_path, train_file_name), sep='\t', index=False, encoding="utf-8", header=0)
    testdf.to_csv(Path().join(output_path, test_file_name), sep='\t', index=False, encoding="utf-8", header=0)


def trans_to_detail(labs, path):
    """ 生成所有级别的商品标签以及相应的数量和位置信息

        生成的内容用于数据分析

        输出格式

            0. 本地生活 350 [0, 349]
                0. 游戏充值 350 [0, 349]
                        0. QQ充值 41 [0, 40]		1. 游戏点卡 309 [41, 349]

            1. 宠物生活 2268 [350, 2617]
                0. 宠物零食 64 [350, 413]
                        0. 猫零食 19 [350, 368]		1. 磨牙/洁齿 45 [369, 413]
                            ··· ···

        :param labs:    按序的完整的商品标签数据 NdArray类型
        :return detail: 所有级别的商品标签以及相应的数量和位置信息
        """

    detail = ""

    fi_s = -1
    fi_e = -1
    fi_cou = -1
    while fi_e != len(labs) - 1:

        detail = detail + '\n'

        fi_s = fi_e + 1
        fi_lab_bas = labs[fi_s].split('--')[0]
        for i in range(fi_s, len(labs)):
            fi_lab = labs[i].split('--')[0]
            if fi_lab != fi_lab_bas:
                fi_e = i - 1
                fi_cou += 1
                detail = detail + str(fi_cou) + '. ' + fi_lab_bas + ' ' + str(fi_e - fi_s + 1) + \
                         ' [' + str(fi_s) + ', ' + str(fi_e) + ']\n'
                break
            if i == len(labs) - 1:
                fi_e = i
                fi_cou += 1
                detail = detail + str(fi_cou) + '. ' + fi_lab_bas + ' ' + str(fi_e - fi_s + 1) + \
                         ' [' + str(fi_s) + ', ' + str(fi_e) + ']\n'
                break

        se_s = fi_s - 1
        se_e = fi_s - 1
        se_cou = -1
        while se_e != fi_e:
            se_s = se_e + 1
            se_lab_bas = labs[se_s].split('--')[1]
            for j in range(se_s, len(labs)):
                se_lab = labs[j].split('--')[1]
                if se_lab != se_lab_bas:
                    se_e = j - 1
                    se_cou += 1
                    detail = detail + '\t' + str(se_cou) + '. ' + se_lab_bas + ' ' + str(se_e - se_s + 1) + \
                             ' [' + str(se_s) + ', ' + str(se_e) + ']\n'
                    break
                if j == len(labs) - 1:
                    se_e = j
                    se_cou += 1
                    detail = detail + '\t' + str(se_cou) + '. ' + se_lab_bas + ' ' + str(se_e - se_s + 1) + \
                             ' [' + str(se_s) + ', ' + str(se_e) + ']\n'
                    break

            detail = detail + '\t\t\t'

            th_s = se_s - 1
            th_e = se_s - 1
            th_cou = -1
            while th_e != se_e:
                th_s = th_e + 1
                th_lab_bas = labs[th_s].split('--')[2]
                for k in range(th_s, len(labs)):
                    th_lab = labs[k].split('--')[2]
                    if th_lab != th_lab_bas:
                        th_e = k - 1
                        th_cou += 1
                        detail = detail + str(th_cou) + '. ' + th_lab_bas + ' ' + str(th_e - th_s + 1) + \
                                 ' [' + str(th_s) + ', ' + str(th_e) + ']\t\t'
                        if (th_cou + 1) % 2 == 0:
                            detail += '\n\t\t\t'
                        break
                    if k == len(labs) - 1:
                        th_e = k
                        th_cou += 1
                        detail = detail + str(th_cou) + '. ' + th_lab_bas + ' ' + str(th_e - th_s + 1) + \
                                 ' [' + str(th_s) + ', ' + str(th_e) + ']\t\t'
                        break

            detail = detail + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(detail)
    return detail
