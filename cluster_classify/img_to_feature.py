import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from utils.image_precess import *
import numpy as np
import cv2
import json


def get_image(img_dir, img_name):
    # cv2 default (H, W)
    return preprocess_input(img_2_array(cv2.resize(cv2.imread(img_dir + img_name), (80, 240))))


def get_feature(img_name):
    img_dir = '/home/ubuntu/media/File/1Various/Person_reid_dataset/DukeMTMC-reID/bounding_box_train/'

    with tf.name_scope('input'):
        image_input = tf.placeholder(tf.float32, [None, 240, 80, 3], name='img_input', )

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        feature, _ = nets.resnet_v1.resnet_v1_50(image_input, is_training=False, global_pool=False, reuse=tf.AUTO_REUSE)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, './model_p_n_1_3_49950.ckpt')
    img_array = get_image(img_dir, img_name)
    print(img_name)
    _feature = sess.run(feature, feed_dict={image_input: np.reshape(img_array, [1, 240, 80, 3])})
    sess.close()
    return _feature


if __name__ == '__main__':
    feature = get_feature('0001_c2_f0046182.jpg')
    # (1, 8, 3, 2048)
    print(feature.shape)