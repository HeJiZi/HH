import os
import random
import tensorflow as tf
import collections
import numpy as np

# def load_w2c_textcn_dataset(path='./data/'):
#     """
#     Returns
#     --------
#     word_list_all : a list
#         a list of string (word).\n
#      要求：中文语料需要先分词
#     """
#
#     print("Load or Download chinese text corpus Dataset> {}".format(path))
#
#     filename = 'wiki_cn.cut'
#     word_list_all = []
#     with open(os.path.join(path, filename)) as f:
#         for line in f:
#             word_list = line.strip().split()
#             for idx, word in enumerate(word_list):
#                 word_list[idx] = word_list[idx].decode('utf-8')
#                 # print word_list[idx]
#                 word_list_all.append(word_list[idx])
#     return word_list_all
#
#
# # words = load_w2c_textcn_dataset(path='./data/')
# # print(len(words))
#
# import collections
#
# vocabulary_size = 200000
# count = [['UNK', -1]]
# count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
# dictionary = dict()
#
# for word, _ in count:
#     dictionary[word] = len(dictionary)
# data = list()
# unk_count = 0
# for word in words:
#     if word in dictionary:
#         index = dictionary[word]
#     else:
#         index = 0  # dictionary['UNK']
#         unk_count = unk_count + 1
#     data.append(index)
#
# count[0][1] = unk_count
# reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
# del words
#
# data_index = 0
#
#
# # batch生成器
# def generate_batch(batch_size, num_skips, skip_window):
#     global data_index
#     batch = np.ndarray(shape=(batch_size), dtype=np.int32) # 生成1*batch_size的向量
#     labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32) # 生成batch_size*1 的向量
#     span = 2 * skip_window + 1  # [ skip_window target skip_window ]
#     buf = collections.deque(maxlen=span)
#     for _ in range(span):
#         buf.append(data[data_index])
#         data_index = (data_index + 1) % len(data)
#     for i in range(batch_size // num_skips):
#         target = skip_window  # target label at the center of the buffer
#         targets_to_avoid = [skip_window]
#         for j in range(num_skips):
#             while target in targets_to_avoid:
#                 target = random.randint(0, span - 1)
#             targets_to_avoid.append(target)
#             batch[i * num_skips + j] = buf[skip_window]
#             labels[i * num_skips + j, 0] = buf[target]
#         buf.append(data[data_index])
#         data_index = (data_index + 1) % len(data)
#     return batch, labels


# batch_size = 128
# embedding_size = 128  # 生成向量维度.
# skip_window = 2  # 左右窗口.
# num_skips = 2  # 同一个keyword产生label的次数.
# num_sampled = 64  # 负样本抽样数.
#
# graph = tf.Graph()
#
# with graph.as_default(), tf.device('/cpu:0'):
#     train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
#     train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
#
#     embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
#     softmax_weights = tf.Variable(
#         tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
#     softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
#
#     embed = tf.nn.embedding_lookup(embeddings, train_dataset)
#     loss = tf.reduce_mean(
#         tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
#                                    labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
#
#     optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
#
#     norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
#     normalized_embeddings = embeddings / norm
from temp.TensorFlowPreprocess import TensorFlowPreprocess
from utils.PathUtil import Path

vocabulary_size = 2000000
embedding_size = 50
label_size = 100
learning_rate = 0.1


def conv_net(x):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], 0, 1.0))
    weight_matrix = tf.Variable(tf.truncated_normal([embedding_size, label_size], stddev=1.0 / np.sqrt(label_size)))
    feature = tf.nn.embedding_lookup(embeddings, x)
    state = tf.reduce_sum(feature, axis=0)
    output = tf.matmul(tf.reshape(state, [1, 5]), weight_matrix)
    return output


def train():
    X = tf.placeholder(tf.float32, [None, embedding_size])
    Y = tf.placeholder(tf.int16, [1, label_size])
    output = conv_net(X)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=Y,
            logits=output
        )
    )
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(1, num_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                loss_value, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})


# def train_single()

def start():
    path = Path()
    tp = TensorFlowPreprocess(path.ori_data)
    X, Y = tp.compile()
    global label_size
    label_size = tp.label_dict.__len__()


if __name__ == '__main__':
    start()
    # sess = tf.Session()
    # embeddings = tf.Variable(tf.random_uniform([3, 5], 0, 1.0))
    # weight_matrix = tf.Variable(tf.truncated_normal([5, 3], stddev=1.0 / np.sqrt(3)))
    # # softmax_weights = tf.Variable(
    # #             tf.truncated_normal([3, 5], stddev=1.0 / np.sqrt(5)))
    # # softmax_biases = tf.Variable(tf.zeros([3]))
    # feature = tf.nn.embedding_lookup(embeddings, [0,1])
    # state = tf.reduce_sum(feature, axis=0)
    # output = tf.matmul(tf.reshape(state,[1,5]),weight_matrix)
    # # prob = tf.nn.softmax(output)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[1], logits=output))
    # optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(optimizer))
    # sess.close()
