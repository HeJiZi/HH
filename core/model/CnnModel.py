from keras import layers
from keras import models
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical


class CnnModel:
    _name = 'cnn'
    _epochs = 3,
    _batch_size = 256
    _model = None

    def __init__(self, input_length, word_num, type_num, epochs=3, batch_size=256, metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        self._epochs = epochs
        self._batch_size = batch_size
        self._init_model(input_length, word_num, type_num, metrics)

    def _init_model(self, input_length, word_num, type_num, metrics):
        self._model = models.Sequential()
        self._model.add(Embedding(word_num, 128, input_length=input_length))
        self._model.add(layers.Conv1D(128, 3, activation='relu'))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(64, activation="relu"))
        self._model.add(layers.Dense(type_num, activation='softmax'))
        self._model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=metrics)
        print(self._model.summary())

    def start_train(self, feature, label):
        """
        输入数据,进行训练
        :param feature: 样本特征 nparray 类型
        :param label: 样本标签 一维的nparry 类型
        :return:
        """
        onehot_label = to_categorical(label)
        self._model.fit(feature, onehot_label)

    def test(self, feature, label):
        """
        输入测试数据，返回测试结果
        :param feature: 样本特征 nparray 类型
        :param label: 样本标签 一维的nparry 类型
        :return: loss,metrics
        """
        onehot_label = to_categorical(label)
        return self._model.evaluate(feature, onehot_label)
