import cv2
import tensorflow as tf
import numpy as np
from im_transf_net import create_net


def gen_model(args_styleTransfer, sess):
    model_path = args_styleTransfer['model_path']
    upsample_method = args_styleTransfer['upsample_method']

    # Create the graph.
    with tf.variable_scope('img_t_net', reuse=tf.AUTO_REUSE):
        X = tf.placeholder(
            tf.float32,
            shape=args_styleTransfer["content_target_resize"],
            name='input'
        )
        Y = create_net(X, upsample_method)

    saver = tf.train.Saver()
    # with tf.Session() as sess:
    saver.restore(sess, model_path)
    return saver


def styleTransfer(frame, args_styleTransfer, saver, sess):
    model_path = args_styleTransfer['model_path']
    upsample_method = args_styleTransfer['upsample_method']
    content_target_resize = args_styleTransfer['content_target_resize']

    # Read + preprocess input image.
    img = cv2.resize(
        frame,
        args_styleTransfer["content_target_resize"],
        interpolation=cv2.INTER_CUBIC,
    )
    img_4d = img[np.newaxis, :]

    # Create the graph.
    with tf.variable_scope('img_t_net', reuse=tf.AUTO_REUSE):
        X = tf.placeholder(
            tf.float32,
            shape=args_styleTransfer["content_target_resize"],
            name='input',
        )
        Y = create_net(X, upsample_method)

    img_out = sess.run(Y, feed_dict={X: img_4d})

    # Postprocess + save the output image.
    img_out = cv2.cvtColor(
        np.squeeze(img_out).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    return img_out
