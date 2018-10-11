import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from utils.image_precess import *
import cv2
import json


def get_image():
    # cv2 default (H, W)
    return preprocess_input(img_2_array(cv2.resize(cv2.imread(img_dir + img_name), (80, 240))))


if __name__ == '__main__':
    file_dir = '../data/DukeMTMC/bounding_box_train.txt'
    img_dir = '/home/ubuntu/media/File/1Various/Person_reid_dataset/DukeMTMC-reID/bounding_box_train/'

    with tf.name_scope('input'):
        image_input = tf.placeholder(tf.float32, [None, 240, 80, 3], name='img_input')

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        feature, _ = nets.resnet_v1.resnet_v1_50(image_input, is_training=False, global_pool=False)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './model_p_n_1_3_49950.ckpt')
        _feature_dict = {}
        with open(file_dir, 'r') as f:
            for i, line in enumerate(f):
                if i > 5:
                    break
                img_name = line[:-1]
                img_array = get_image()
                print(img_name)
                _feature = sess.run(feature, feed_dict={image_input: np.reshape(img_array, [1, 240, 80, 3])})
                _feature_dict[img_name] = _feature.tolist()
                # (1, 8, 3, 2048)
                # print([_feature])
        with open('./img_feature.json', 'w') as w:
            json.dump(_feature_dict, w)

    # 0001
    # _c2_f0046182.jpg
    # 0001
    # _c2_f0046302.jpg
    # 0001
    # _c2_f0046422.jpg
    # 0001
    # _c2_f0046542.jpg
    # 0001
    # _c2_f0046662.jpg
    # 0001
    # _c2_f0046782.jpg

    with open('./img_feature.json', 'r') as f:
        for i, ii in enumerate(f):
            print(i)
        # a = json.load(f)
        # for i in a:
        #     print(i)