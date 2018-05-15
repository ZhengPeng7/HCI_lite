import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import extract_chars
from collections import OrderedDict
import os
from matplotlib.font_manager import FontProperties
import sys


def predict_formula(characs):
    characters_ref = OrderedDict().fromkeys([
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '+', '-', 'x', '/'
        ])
    characters_ref_keys = list(characters_ref.keys())
    # y_train = []
    # x_train = []
    y_test = np.zeros((len(characs), 14), dtype=np.uint8)
    x_test = np.array(characs)
    # print("y_test.shape={}".format(y_test.shape))
    # print("x_test.shape={}".format(x_test.shape))

    class_num = y_test.shape[-1]
    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, class_num])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    conv1_weights = tf.get_variable(
        "conv1_weights",
        [5, 5, 1, 32],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv1_biases = tf.get_variable("conv1_biases", [32],
                                   initializer=tf.constant_initializer(0.0))
    conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1],
                         padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    conv2_weights = tf.get_variable(
        "conv2_weights",
        [5, 5, 32, 64],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv2_biases = tf.get_variable(
        "conv2_biases", [64], initializer=tf.constant_initializer(0.0))
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1],
                         padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024],
                                  initializer=tf.truncated_normal_initializer(
                                  stddev=0.1))
    fc1_biases = tf.get_variable(
        "fc1_biases", [1024], initializer=tf.constant_initializer(0.1))
    pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_biases)
    keep_prob = tf.placeholder(tf.float32)
    fc1_dropout = tf.nn.dropout(fc1, keep_prob)
    fc2_weights = tf.get_variable("fc2_weights", [1024, class_num],
                                  initializer=tf.truncated_normal_initializer(
                                  stddev=0.1))
    fc2_biases = tf.get_variable(
        "fc2_biases", [class_num], initializer=tf.constant_initializer(0.1))
    fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases
    y_conv = tf.nn.softmax(fc2)
    pred_class_index = tf.argmax(y_conv, 1)
    # restore well-trained model
    saver = tf.train.Saver()
    saver.restore(sess, './my_model/model.ckpt')
    batch_size_test = 1
    if not y_test.shape[0] % batch_size_test:
        epoch_test = y_test.shape[0] // batch_size_test
    else:
        epoch_test = y_test.shape[0] // batch_size_test + 1
    pred_values = []
    for i in range(epoch_test):
        if (i*batch_size_test % x_test.shape[0]) > (((i+1)*batch_size_test) %
                                                    x_test.shape[0]):
            x_data_test = np.vstack((
                x_test[i*batch_size_test % x_test.shape[0]:],
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
        pred_value = to_categorical(np.squeeze(
            sess.run([pred_class_index], feed_dict={
                   x: x_data_test, y_: y_data_test, keep_prob: 1.0})), 68)
        pred_values.append(characters_ref_keys[(np.argmax(pred_value))])
    return pred_values


def main():

    # matplotlib 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # myfont = FontProperties(fname='/usr/share/fonts/truetype/simhei.ttf', size=20)

    image = sys.argv[1]
    characs = extract_chars.extract_chars(image)
    pred_values = []
    # for charac in characs:
    pred_values.append(predict_formula(characs))
    print('formula is:', pred_values)


if __name__ == '__main__':
    main()