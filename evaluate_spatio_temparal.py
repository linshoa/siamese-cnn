import utils.cuda_util as cuda
import utils.config as config
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from utils.image_precess import *
import numpy as np
from utils.extract_batch import *
import cv2

# model_siamese_cnn = './model/spatio_temparal_visual_result/model_p_n_1_3_49900.ckpt'
model_siamese_cnn = './cluster_classify/model_p_n_1_3_49950.ckpt'
query_img_dir = config.DukeMTMC_img_dir+'query/'
test_img_dir = config.DukeMTMC_img_dir+'bounding_box_test/'
query_name_dir = config.DukeMTMC_name_dir+'query.txt'
test_name_dir = config.DukeMTMC_name_dir+'bounding_box_test.txt'

config.BATCH_SIZE = 1


def spatio_temporal(location_left, location_right, time_left, time_right):
    spatio_diff = tf.abs(time_left-time_right)
    location_diff = tf.abs(location_left-location_right)
    diff_concat = tf.concat([spatio_diff, location_diff], axis=1)
    fc_1 = slim.fully_connected(slim.flatten(diff_concat), 16)
    fc_reshape = tf.reshape(fc_1, [config.BATCH_SIZE, 1, 16])
    _all = {'spatio_diff': spatio_diff, 'location_diff': location_diff, 'diff_concat': diff_concat}
    return fc_reshape, _all


def contrastive_loss(_left, _right, _info_left_location, _info_right_location, _info_left_time, _info_right_time):
    with tf.name_scope('output'):
        # the transpose here really memory consume!!!
        # inner_product = tf.matmul(_left, tf.matrix_transpose(_right))
        # todo drpout may not need.
        _left_l2 = tf.nn.l2_normalize(_left, name='left_l2_norm')
        _right_l2 = tf.nn.l2_normalize(_right, name='right_l2_norm')
        diff_feature = tf.matmul(_left_l2, _right_l2, transpose_a=False, transpose_b=True)
        fc_out = slim.fully_connected(slim.flatten(diff_feature), 16)
        _inner_product = tf.reshape(fc_out, [config.BATCH_SIZE, 1, 16])

        feature_spatio_temporal, _all = spatio_temporal(_info_left_location, _info_right_location, _info_left_time, _info_right_time)

        concat_visual_spatio_temporal = tf.concat([_inner_product, feature_spatio_temporal], axis=1)

        final_feature = slim.fully_connected(slim.flatten(concat_visual_spatio_temporal), 2, activation_fn=None)
        final_feature_out = tf.reshape(final_feature, [config.BATCH_SIZE, 2])
        # 输出 [1,0] or [0, 1]
        return final_feature_out


def get_id_location_time(line):
    data = line.split('_')
    probe_id = data[0]
    probe_location = data[1][1]
    probe_time = data[2][1:-5]
    return probe_id, probe_location, probe_time


def to_show_array(_end_):
    print(_end_['resnet_v1_50/block1'].shape)
    left_img = array_to_img(np.resize(_end_['resnet_v1_50/block1'][:, :, :, :6], [224, 80, 3]))
    left_img.show()
    left_img.close()


with tf.Session(config=cuda.config) as sess:
    with tf.name_scope('input'):
        left = tf.placeholder(tf.float32, [None, 240, 80, 3], name='left')
        right = tf.placeholder(tf.float32, [None, 240, 80, 3], name='right')
        left_location = tf.placeholder(tf.float32, [None, 1], name='left_location')
        right_location = tf.placeholder(tf.float32, [None, 1], name='right_location')
        left_time = tf.placeholder(tf.float32, [None, 1], name='left_time')
        right_time = tf.placeholder(tf.float32, [None, 1], name='right_time')

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        # todo is_training True
        # for the paper of vehicle, global_pool is True
        left_feature, _ = nets.resnet_v1.resnet_v1_50(left, is_training=True, global_pool=False, reuse=False)
        right_feature, __ = nets.resnet_v1.resnet_v1_50(right, is_training=True, global_pool=False, reuse=True)

    final_out = contrastive_loss(left_feature, right_feature, left_location, right_location, left_time, right_time)

    restore = tf.train.Saver()
    restore.restore(sess, model_siamese_cnn)
    with open(query_name_dir, 'r') as query_name:
        for query_line in query_name:
            _query_time = []
            query_id, query_location, query_time = get_id_location_time(query_line)
            _query_time.append(query_time)
            print(query_id)
            with open(test_name_dir, 'r') as test_name:
                for test_line in test_name:
                    _test_time = []
                    test_id, test_location, test_time = get_id_location_time(test_line)
                    _test_time.append(test_time)
                    _query_array = preprocess_input(img_2_array(load_img(query_img_dir+query_line[:-1], target_size=[240, 80])))
                    _test_array = preprocess_input(img_2_array(load_img(test_img_dir+test_line[:-1], target_size=[240, 80])))

                    _final_score, _left_feature, _right_feature = sess.run([final_out, left_feature, right_feature], feed_dict={left: [_query_array], right: [_test_array], left_location: [[query_location]], right_location: [[test_location]], left_time: [_query_time], right_time: [_test_time]})
                    print(test_id, _final_score)
                    if np.argmax(_final_score[0]) == 0:
                        print(test_id)
