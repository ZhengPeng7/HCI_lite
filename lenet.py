import tensorflow as tf
import tfrecords2array
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import sys


def lenet(epoch_train):

    y_train = []
    x_train = []
    y_test = []
    x_test = []
    train_data = tfrecords2array.tfrecord2array(
        os.path.join("data_tfrecords", "train.tfrecords")
    )
    test_data = tfrecords2array.tfrecord2array(
        os.path.join("data_tfrecords", "test.tfrecords")
    )
    y_train.append(train_data[0])
    x_train.append(train_data[1])
    y_test.append(test_data[0])
    x_test.append(test_data[1])
    for i in [y_train, x_train, y_test, x_test]:
        for j in i:
            print(j.shape)
    y_train = np.vstack(y_train)
    x_train = np.vstack(x_train)
    y_test = np.vstack(y_test)
    x_test = np.vstack(x_test)

    class_num = y_test.shape[-1]

    print("x_train.shape=" + str(x_train.shape))
    print("x_test.shape=" + str(x_test.shape))
    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, class_num])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # layer 1-st, convolutional
    conv1_weights = tf.get_variable(
        "conv1_weights",
        [5, 5, 1, 32],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv1_biases = tf.get_variable("conv1_biases", [32],
                                   initializer=tf.constant_initializer(0.0))
    conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1],
                         padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # layer 2-nd, max-pooling
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

    # layer 3-rd, convolutional
    conv2_weights = tf.get_variable(
        "conv2_weights",
        [5, 5, 32, 64],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv2_biases = tf.get_variable(
        "conv2_biases", [64], initializer=tf.constant_initializer(0.0))
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1],
                         padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # layer 4-th, max-pooling
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

    # layer 5-th, FC
    fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024],
                                  initializer=tf.truncated_normal_initializer(
                                  stddev=0.1))
    fc1_biases = tf.get_variable(
        "fc1_biases", [1024], initializer=tf.constant_initializer(0.1))
    pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_biases)

    keep_prob = tf.placeholder(tf.float32)
    fc1_dropout = tf.nn.dropout(fc1, keep_prob)

    # layer 6-th, FC
    fc2_weights = tf.get_variable("fc2_weights", [1024, class_num],
                                  initializer=tf.truncated_normal_initializer(
                                  stddev=0.1))
    fc2_biases = tf.get_variable(
        "fc2_biases", [class_num], initializer=tf.constant_initializer(0.1))
    fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases

    # layer 7-th, output
    y_conv = tf.nn.softmax(fc2)

    # define cross-entropy loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                                  reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training start
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    acc_train_train = []
    acc_train_test = []
    batch_size = 64
    # epoch_train = 2**14     # restricted by the hardware in my computer
    print("Training steps=" + str(epoch_train))
    for i in range(epoch_train):
        if (i*batch_size % x_train.shape[0]) > ((i + 1)*batch_size %
                                                x_train.shape[0]):
            x_data_train = np.vstack(
                (x_train[i*batch_size % x_train.shape[0]:],
                 x_train[:(i+1)*batch_size % x_train.shape[0]]))
            y_data_train = np.vstack(
                (y_train[i*batch_size % y_train.shape[0]:],
                 y_train[:(i+1)*batch_size % y_train.shape[0]]))
            x_data_test = np.vstack(
                (x_test[i*batch_size % x_test.shape[0]:],
                 x_test[:(i+1)*batch_size % x_test.shape[0]]))
            y_data_test = np.vstack(
                (y_test[i*batch_size % y_test.shape[0]:],
                 y_test[:(i+1)*batch_size % y_test.shape[0]]))
        else:
            x_data_train = x_train[
                i*batch_size % x_train.shape[0]:
                (i+1)*batch_size % x_train.shape[0]]
            y_data_train = y_train[
                i*batch_size % y_train.shape[0]:
                (i+1)*batch_size % y_train.shape[0]]
            x_data_test = x_test[
                i*batch_size % x_test.shape[0]:
                (i+1)*batch_size % x_test.shape[0]]
            y_data_test = y_test[
                i*batch_size % y_test.shape[0]:
                (i+1)*batch_size % y_test.shape[0]]
        if i % 640 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={x: x_data_train, y_: y_data_train, keep_prob: 1.0})
            test_accuracy = accuracy.eval(
                feed_dict={x: x_data_test, y_: y_data_test, keep_prob: 1.0})
            print("step {}, training accuracy={}, testing accuracy={}".format(
                i, train_accuracy, test_accuracy))
            acc_train_train.append(train_accuracy)
            acc_train_test.append(test_accuracy)
        train_step.run(feed_dict={
            x: x_data_train, y_: y_data_train, keep_prob: 0.5})
    print("saving model...")
    save_path = saver.save(sess, "./models/lenet.ckpt")
    print("save model:{0} Finished".format(save_path))

    batch_size_test = 64
    epoch_test = y_test.shape[0] // batch_size_test + 1
    acc_test = 0
    for i in range(epoch_test):
        if (i*batch_size_test % x_test.shape[0]) > ((i + 1)*batch_size_test %
                                                    x_test.shape[0]):
            x_data_test = np.vstack((
                x_test[i*batch_size_test % x_train.shape[0]:],
                x_test[:(i+1)*batch_size_test % x_test.shape[0]]))
            y_data_test = np.vstack((
                y_test[i*batch_size_test % y_test.shape[0]:],
                y_test[:(i+1)*batch_size_test % y_test.shape[0]]))
        else:
            x_data_test = x_test[
                i*batch_size_test % x_test.shape[0]:
                (i+1)*batch_size_test % x_test.shape[0]]
            y_data_test = y_test[
                i*batch_size_test % y_test.shape[0]:
                (i+1)*batch_size_test % y_test.shape[0]]

        # Calculate batch loss and accuracy
        c = accuracy.eval(feed_dict={
            x: x_data_test, y_: y_data_test, keep_prob: 1.0})
        acc_test += c / epoch_test
        print("{}-th test accuracy={}".format(i, acc_test))
    print("At last, test accuracy={}".format(acc_test))

    print("Finish!")
    return acc_train_train, acc_train_test, acc_test


def plot_acc(acc_train_train, acc_train_test, acc_test):
    plt.figure(1)
    p1, p2 = plt.plot(list(range(len(acc_train_train))),
                      acc_train_train, 'r-',
                      list(range(len(acc_train_test))),
                      acc_train_test, 'b-')
    plt.legend(handles=[p1, p2], labels=["training_acc", "testing_acc"])
    plt.title("Accuracies During Training")
    plt.show()


def main():
    num_epoch_training = int(sys.argv[1]) if sys.argv[1] else 2 ** 15
    acc_train_train, acc_train_test, acc_test = lenet(num_epoch_training)
    plot_acc(acc_train_train, acc_train_test, acc_test)


if __name__ == '__main__':
    main()