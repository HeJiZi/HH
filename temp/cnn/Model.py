from keras import layers
from keras import models
from keras.models import load_model
import os


class CNN:

    def __init__(self):
        self.model = models.Sequential()

    def cnn_module(self, tra_charas, tes_charas, tra_y, tes_y, out_len):
        self.model.add(layers.Conv1D(32, 4, activation='relu', input_shape=(20, 128)))  # 125
        self.model.add(layers.MaxPool1D(2))    # 63
        self.model.add(layers.Conv1D(64, 4, activation='relu'))   # 60
        self.model.add(layers.MaxPool1D(2))    # 30
        self.model.add(layers.Conv1D(64, 4, activation='relu'))   # 27
        self.model.add(layers.MaxPool1D(2))    # 14
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(out_len, activation='softmax'))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(tra_charas, tra_y, epochs=5, batch_size=64)
        test_loss, test_acc = self.model.evaluate(tes_charas, tes_y)
        print("test loss is : ", test_loss)
        print("test accuracy is : ", test_acc)
        return test_acc

    def cnn_predict(self, val_charas):
        return self.model.predict(val_charas)

    def cnn_save(self, save_name):
        model_path = './model/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model.save(os.path.join(model_path, save_name + '.h5'))

    def load(self, model_path):
        self.model = load_model(model_path)
