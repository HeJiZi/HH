from core.preprocess.Preprocess import Preprocess
from core.model.CnnModel import CnnModel
from utils.PathUtil import Path
from sklearn.model_selection import train_test_split
import pickle

if __name__ == '__main__':
    path = Path()
    # pre = Preprocess(path.ori_data, level=1)

    pre = Preprocess(path.ori_data, level=3)
    # df = pre.compile()

    # pre.save(path.vectors, path.word_code, path.type_code)
    df = pre.load(path.vectors, path.word_code, path.type_code)

    # x_train, x_test, y_train, y_test = train_test_split(df.ix[:, 1:], df.ix[:, 0], test_size=.2, random_state=520)
    from sklearn.utils import shuffle
    df = shuffle(df)
    data = []
    # with open("../data/acc.csv", "w", encoding="utf-8") as f:
    #     f.write('损失,成功率,轮询'+'\n')
    cnn = CnnModel(df.shape[1] - 1, pre.word_size, pre.type_size, epochs=40)
    hs = cnn.start_train(df.ix[:, 1:].values, df.ix[:, 0].values)
    # loss, acc = cnn.test(x_test.values, y_test.values)
        # data.append()
    with open("../data/acc.csv", "wb") as f:
        pickle.dump(hs, f)

    # print(acc)

        # for da in data:
        #     f.write(",".join(da)+'\n')

