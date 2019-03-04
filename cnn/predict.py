import pandas as pd
import numpy as np
from utils.jiebaExtend import jiebaExtend
from utils.WordVec import SpacyVec
from Model import CNN

if __name__ == '__main__':
    cnn = CNN()
    tra_ori_path = "./data/train.tsv"
    # 读取数据
    ori_data_df = pd.read_csv(tra_ori_path, sep='\t', encoding='gb18030', nrows=None)
    # 转 DataFrame 为 NdArray
    ori_data = np.array(ori_data_df)
    # 打乱数据
    np.random.shuffle(ori_data)

    group_size = 500000
    trai = 0.7
    test = 0.25
    vail = 0.05

    sams = ori_data[0: group_size, 0]
    labs = ori_data[0: group_size, 1]


    # 验证集数据处理
    print("开始处理验证数据")
    vai_sams = sams[int((trai + test) * len(sams)):len(sams)]
    vai_labs = labs[int((trai + test) * len(labs)):len(labs)]

    vai_toks = jiebaExtend.cut_to_array(vai_sams)
    vai_charas = SpacyVec.array_to_vec(vai_toks)


    print("验证数据处理完成")
    cnn.load("my_model.h5")
    res = cnn.cnn_predict(vai_charas)

    pre = []
    for i in range(len(res)):
        #   print("预测的各个种类的概率：")
        # print(res[i])
        # print("最大概率为：", max(res[i]))
        for key, value in enumerate(res[i]):
            if max(res[i]) == value:
                # print("商品种类下标: ", key)
                pre.append(key)
    print(pre)
    # 2.5万个数据做验证正确率为 0.81
