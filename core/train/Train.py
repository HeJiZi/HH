from core.preprocess.Preprocess import Preprocess
from utils.PathUtil import Path
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras import layers
from keras import models
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

p = Path()
pre = Preprocess(p.ori_data, level=1)
arry = pre.compile()
df = pd.DataFrame(arry).fillna(0).astype(np.int32)
df.to_csv(p.save_directory+'/vectors.csv', sep=',', index=False, encoding="utf-8", header=0)
pre.save_type_dic(p.type_code)
pre.save_word_dic(p.word_code)
# df = pd.read_csv(p.save_directory+'/vectors.csv', sep=',', header=None, encoding="utf-8", nrows=None)

# 构建顺序模型
model = models.Sequential()

# # 添加一层卷积层
# # 32:filter(滤波器)个数 ,(3,3)设定滤波器大小为3*3, 设定激活函数为relu,input_shape为输入空间的维数)
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#
# # 添加池化层,大小为2*2
# model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# #model.add(layers.MaxPool2D((2, 2)))
# #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(Embedding(pre.word_size, 128, input_length=df.shape[1]-1))
# model.add(Embedding(294525, 128, input_length=df.shape[1]-1))
# 添加展平层
model.add(layers.Flatten())

# 添加全连接层
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(pre.type_size, activation='softmax'))
# model.add(layers.Dense(22, activation='softmax'))
# 编译模型,metrics:评估值,这里设为正确率
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(df.ix[:, 1:], df.ix[:, 0], test_size=.2, random_state=520)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 训练模型
model.fit(x_train.values, y_train, epochs=10, batch_size=256)


# 用训练集测试模型
test_loss, test_acc = model.evaluate(x_test.values, y_test)
#
print("test_acc is ")
print(test_acc)


