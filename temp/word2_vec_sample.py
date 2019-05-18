import tensorflow as tf
import numpy as np
import time

from sklearn.model_selection import train_test_split

from temp.TensorFlowPreprocess import TensorFlowPreprocess
from utils.PathUtil import Path

vocabulary_size = 200000
max_num_in_line = 50
embedding_size = 50
label_size = 100
learning_rate = 0.01
batch_size = 32
test_batch_size = 10000
epochs = 60
print_step = 6
gpu_num = 2


def conv_net(x):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], 0, 1.0))
    weight_matrix = tf.Variable(tf.truncated_normal([embedding_size, label_size], stddev=1.0 / np.sqrt(label_size)))
    feature = tf.nn.embedding_lookup(embeddings, x)
    state = tf.reduce_sum(feature, axis=1)
    output = tf.matmul(state, weight_matrix)
    return output


def transform_feature(from_data, word_dict):
    length = len(from_data)
    result = np.zeros([length, max_num_in_line])
    for i in range(length):
        for j in range(max_num_in_line):
            if j >= len(from_data[i]):
                break
            result[i, j] = 1 + word_dict[from_data[i][j]]  # 编号为0代表没有单词，+1跳过该编号
    return result


def to_one_hot(labels, label_num):
    length = len(labels)
    result = np.zeros([length, label_num])
    for i in range(length):
        result[i, labels[i]] = 1
    return result


def compute_feed_time(data_num, batch_size):
    temp = data_num // batch_size
    return temp + 1 if data_num % batch_size != 0 else temp


def train_single():
    path = Path()
    tp = TensorFlowPreprocess(path.ori_data)
    names, types = tp.compile()
    train_names, test_names, train_types, test_types = train_test_split(
        names, types, test_size=.2, random_state=520)

    global label_size
    label_size = tp.label_dict.__len__()

    X = tf.placeholder(tf.int32, [None, None])
    Y = tf.placeholder(tf.float32, [None, label_size])
    output = conv_net(X)
    probs = tf.nn.softmax(output)

    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(probs), reduction_indices=[1]))
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(probs, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_name_len = len(train_names)
    test_name_len = len(test_names)
    test_x = transform_feature(test_names, tp.word_dict)
    test_y = to_one_hot([tp.label_dict[label] for label in test_types], label_size)

    train_times = compute_feed_time(train_name_len, batch_size)
    test_times = compute_feed_time(test_name_len, test_batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for step in range(0, train_times):
                start = step * batch_size
                end = (step + 1) * batch_size if step < train_times - 1 else train_name_len
                batch_x = transform_feature(train_names[start:end], tp.word_dict)
                batch_y = to_one_hot([tp.label_dict[label] for label in train_types[start:end]], label_size)

                sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
                if step % print_step == 0 or step == train_times - 1:
                    loss_val, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
                    print("--已完成[{0}/{1}],loss_val={2:.2f},acc={3:.2f}--".format(
                        end, train_name_len, loss_val, acc))

            test_loss_val = 0
            test_acc = 0
            for step in range(0, test_times):  # 显存不足，分批次feed再求均值以解决问题
                start = step * test_batch_size
                end = (step + 1) * test_batch_size if step < test_times - 1 else test_name_len
                batch_x = test_x[start:end]
                batch_y = test_y[start:end]

                loss_val, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
                test_loss_val += loss_val
                test_acc += acc
                print("--已完成[{0}/{1}]--".format(end, test_name_len))
            print("--已完成[{0}/{1}]轮询，test_loss_val={2:.2f},test_acc={3:.2f}--".format(
                epoch + 1, epochs, test_loss_val / test_times, test_acc / test_times
            ))


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train_parallel():
    path = Path()
    tp = TensorFlowPreprocess(path.ori_data)
    names, types = tp.compile()
    train_names, test_names, train_types, test_types = train_test_split(
        names, types, test_size=.2, random_state=520)

    global label_size
    label_size = tp.label_dict.__len__()

    X = tf.placeholder(tf.int32, [None, None])
    Y = tf.placeholder(tf.float32, [None, label_size])

    tower_grads = []
    with tf.device("/cpu:0"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(gpu_num):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                        _x = X[i * batch_size:(i + 1) * batch_size]
                        _y = Y[i * batch_size:(i + 1) * batch_size]
                        tf.get_variable_scope().reuse_variables()
                        output = conv_net(_x)
                        probs = tf.nn.softmax(output)

                        loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(probs), reduction_indices=[1]))
                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(grads)

    test_output = conv_net(X)
    test_probs = tf.nn.softmax(test_output)
    test_loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(test_probs), reduction_indices=[1]))

    correct_prediction = tf.equal(tf.argmax(test_probs, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_name_len = len(train_names)
    test_name_len = len(test_names)
    test_x = transform_feature(test_names, tp.word_dict)
    test_y = to_one_hot([tp.label_dict[label] for label in test_types], label_size)

    train_times = compute_feed_time(train_name_len, batch_size * gpu_num)
    test_times = compute_feed_time(test_name_len, test_batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for step in range(0, train_times):
                start = step * batch_size
                end = (step + 2) * batch_size if step < train_times - 1 else train_name_len
                batch_x = transform_feature(train_names[start:end], tp.word_dict)
                batch_y = to_one_hot([tp.label_dict[label] for label in train_types[start:end]], label_size)

                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % print_step == 0 or step == train_times - 1:
                    loss_val, acc = sess.run([test_loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
                    print("--已完成[{0}/{1}],loss_val={2:.2f},acc={3:.2f}--".format(
                        end, train_name_len, loss_val, acc))

            test_loss_val = 0
            test_acc = 0
            for step in range(0, test_times):  # 显存不足，分批次feed再求均值以解决问题
                start = step * test_batch_size
                end = (step + 1) * test_batch_size if step < test_times - 1 else test_name_len
                batch_x = test_x[start:end]
                batch_y = test_y[start:end]

                loss_val, acc = sess.run([test_loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
                test_loss_val += loss_val
                test_acc += acc
                print("--已完成[{0}/{1}]--".format(end, test_name_len))
            print("--已完成[{0}/{1}]轮询，test_loss_val={2:.2f},test_acc={3:.2f}--".format(
                epoch + 1, epochs, test_loss_val / test_times, test_acc / test_times
            ))


if __name__ == '__main__':
    train_single()
    # sess = tf.Session()
    #
    # embeddings = tf.Variable(tf.random_uniform([3, 5], 0, 1.0))
    # weight_matrix = tf.Variable(tf.truncated_normal([5, 3], stddev=1.0 / np.sqrt(3)))
    # X = tf.placeholder(tf.int32, [None, None])
    # Y = tf.placeholder(tf.float32, [None, 3])
    # feature = tf.nn.embedding_lookup(embeddings, X)
    # state = tf.reduce_sum(feature, axis=1)
    # output = tf.matmul(state, weight_matrix)
    # probs = tf.nn.softmax(output)
    # loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(probs), reduction_indices=[1]))
    # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    #
    # # grad = optimizer.compute_gradients(loss)
    #
    # sess.run(tf.global_variables_initializer())
    #
    # fe = [[0, 1,1], [1, 2, 1]]
    # la = to_one_hot([1, 1], 3)
    # print(sess.run(loss, feed_dict={X: fe, Y: la}))
    #
    # sess.close()
