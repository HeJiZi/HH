"""
获取数据工具
"""


def get_lab_dict_by_name(labs, name=None, lev=0):
    """ 通过给出的父级标签名称，生成对应的子级标签的字典

    例如给出的 name 为 宠物生活 ， lev 为 1 （由 0 开始计数）

        宠物生活
            宠物零食    宠物玩具    宠物主粮    出行装备    家居日用    洗护美容    医疗保健

    输出为

        dict_lab:
            {'宠物零食': 0, '宠物玩具': 1, '宠物主粮': 2, '出行装备': 3,
             '家居日用': 4, '洗护美容': 5, '医疗保健': 6}
        dict_num:
            {0: '宠物零食', 1: '宠物玩具', 2: '宠物主粮', 3: '出行装备',
            4: '家居日用', 5: '洗护美容', 6: '医疗保健'}

    如果 lev 为 1 或 2 以外的数字，此方法将给出第 0 级商品标签

    :param labs:    按序的完整的商品标签数据
    :param name:    父级标签的名称， lev 参数为 1 或 2 时有效
    :param lev:     需要生成字典的商品标签的级别，即子级标签的级别，默认为 0 ，输出第 0 级的商品标签
    :return dict_lab:   标签--数字 字典，以标签为索引
    :return dict_num:   数字--标签 字典，以数字为索引
    """

    ls = []
    if lev == 1 or lev == 2:
        print("当前生成第 %d 级标签字典（由 0 开始计数）" % lev)
        for lab in labs:
            par_lab = lab.split('--')[lev - 1]
            if name == par_lab:
                lev_lab = lab.split('--')[lev]
                if lev_lab not in ls:
                    ls.append(lev_lab)
    else:
        print("当前生成第 0 级标签字典（由 0 开始计数）")
        for lab in labs:
            lev_lab = lab.split('--')[0]
            if lev_lab not in ls:
                ls.append(lev_lab)

    dict_lab = {}
    dict_num = {}
    for num, lab in enumerate(ls):
        dict_lab[lab] = num
        dict_num[num] = lab

    return dict_lab, dict_num


def get_range_by_name(labs, name=None, lev=0):
    """ 通过给出的标签名称，计算出标签对应的商品信息下标范围

    For example

       给出标签名称 宠物生活 ， lev = 0 ，返回闭区间 [350, 2617]
       给出标签名称 猫零食 ， lev = 2 ，返回闭区间 [350, 368]

    :param labs: 按序的完整的商品标签数据
    :param name: 标签的名称
    :param lev: 需要生成范围的商品标签的级别
    :return range: 对应商品的下标范围，闭区间
    """
    range = []
    last = -1
    for i, lab in enumerate(labs):
        lab_name = lab.split('--')[lev]
        if name == lab_name:
            if last == -1:
                range.append(i)
            last = i
    range.append(last)

    return range

