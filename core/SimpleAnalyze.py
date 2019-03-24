from core.preprocess.Preprocess import Preprocess
from core.model.CnnModel import CnnModel
from core.preprocess.PreprocessPlus import PreprocessPlus
from utils.PathUtil import Path
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    path = Path()
    # pre = Preprocess(path.ori_data, level=1)
    pre = Preprocess(path.ori_data, level=3)
    # df = pre.compile()

    # pre.save(path.vectors, path.word_code, path.type_code)
    df = pre.load(path.vectors, path.word_code, path.type_code)

    x_train, x_test, y_train, y_test = train_test_split(df.ix[:, 1:], df.ix[:, 0], test_size=.2, random_state=520)

    current_epoch = 5
    step = 5
    data = []
    while current_epoch <= 40:
        cnn = CnnModel(df.shape[1] - 1, pre.word_size, pre.type_size, epochs=current_epoch)
        cnn.start_train(x_train.values, y_train.values)
        loss, acc = cnn.test(x_test.values, y_test.values)
        data.append([loss, acc, current_epoch])
        current_epoch += step

    # print(acc)
    with open("../data/acc.csv", "w") as f:
        f.write('损失,成功率,轮询'+'\n')
        for da in data:
            f.write(",".join(da)+'\n')
