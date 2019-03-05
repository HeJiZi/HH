"""
文件写出工具
"""


def export_classes(cls_dic, output_file):
    """
    将类别字典中的内容导出到文件中
    :param cls_dic: 类别字典
    :param output_file: 导出文件路径
    """
    of = open(output_file, "w", encoding="utf-8")
    for className in cls_dic.keys():
        of.write(className + " " + str(cls_dic[className]) + '\n')
    of.close()
