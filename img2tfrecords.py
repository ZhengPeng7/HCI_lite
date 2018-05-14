import numpy as np
import tensorflow as tf
import time
import os
import cv2
from sklearn.utils import shuffle
import sys


# 读取图片
def read_images(path_labels, path_imgs, num_validation=None):
    labels = np.load(path_labels)
    labels = labels.reshape(labels.shape[0], -1)
    imgs = np.load(path_imgs)
    imgs = imgs.reshape(imgs.shape[0], -1)
    if num_validation is None:
        num_validation = labels.shape[0] // 5
    print(labels.shape, imgs.shape)
    data = np.hstack((labels, imgs))
    data = shuffle(data)
    test_labels = data[:num_validation, 0]
    test_images = data[:num_validation, 1:]
    train_labels = data[num_validation:, 0]
    train_images = data[num_validation:, 1:]
    return train_labels, train_images, test_labels, test_images


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(images, labels, filename):
    # 获取要转换为TFRecord文件的图片数目
    num = images.shape[0]
    print("num:", num)
    print("images.shape:", images.shape)
    # 输出TFRecord文件的文件名
    print('Writting', filename)
    # 创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num):
        # 将图像矩阵转化为一个字符串
        img_raw = images[i].tostring()
        # 将一个样例转化为Example Protocol Buffer，并将所有需要的信息写入数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(labels[i])),
            'image_raw': _bytes_feature(img_raw)}))
        # 将example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()
    print('Writting End')


def main():
    # 图片存放位置
    PATH_RES = r'data/'
    PATH_DES = r'data_tfrecords/'
    if len(sys.argv) == 3:
        PATH_RES = sys.argv[1]
        PATH_DES = sys.argv[2]

    start_time = time.time()
    path_images = os.path.join(PATH_RES, "x.npy")
    path_labels = os.path.join(PATH_RES, "y.npy")
    print('Reading images ...'.format(PATH_RES))
    train_labels, train_images, test_labels, test_images = read_images(
        path_labels, path_images
    )
    # Slice data here.
    print('Converting to tfrecords into {} begin'.format(PATH_DES))
    print('\ttrain.tfrecords')
    convert(
        train_images, train_labels, os.path.join(PATH_DES, "train.tfrecords")
    )
    print('\ttest.tfrecords')
    convert(
        test_images, test_labels, os.path.join(PATH_DES, "test.tfrecords")
    )
    duration = time.time() - start_time
    print('Converting end , total cost = %d sec' % duration)


if __name__ == '__main__':
    main()