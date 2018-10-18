import utils.cuda_util as cuda
import utils.config as config
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from utils.image_precess import *
import numpy as np
import time


def count_accuracy_recall(_score_matrix):
    # only [1, 0] is True
    # index is 0 is what we want.
    judge_array = np.argmax(_score_matrix, 2) == 0
    batch_accuracy = np.mean(judge_array)
    for index, one_array in enumerate(judge_array):
        if one_array:
            print('find it : ', right_name[index])
    print(batch_accuracy)
    return batch_accuracy

# todo modify only visual test
model_siamese_cnn = './model/only_visual_result/model_no_dropout_1_3_49600.ckpt'

query_img_dir = config.DukeMTMC_img_dir+'query/'
test_img_dir = config.DukeMTMC_img_dir+'bounding_box_test/'
# train_img_dir = config.DukeMTMC_img_dir+'bounding_box_train/'

query_name_dir = config.DukeMTMC_name_dir+'query.txt'
test_name_dir = config.DukeMTMC_name_dir+'bounding_box_test.txt'
# train_name_dir = config.DukeMTMC_name_dir+'bounding_box_train.txt'

with tf.Session(config=cuda.config) as sess:
    with tf.name_scope('input'):
        left = tf.placeholder(tf.float32, [None, 224, 224, 3], name='left')
        right = tf.placeholder(tf.float32, [None, 224, 224, 3], name='right')

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        # for the paper of vehicle, global_pool is True
        left_feature, _ = nets.resnet_v1.resnet_v1_50(left,  global_pool=False, reuse=False)
        right_feature, __ = nets.resnet_v1.resnet_v1_50(right, global_pool=False, reuse=True)

        """cos"""
        _left = slim.flatten(left_feature)
        _right = slim.flatten(right_feature)
        _left_l2 = tf.nn.l2_normalize(_left, name='left_l2_norm')
        _right_l2 = tf.nn.l2_normalize(_right, name='right_l2_norm')
        diff_feature = tf.multiply(_left_l2, _right_l2)
        fc_out = slim.fully_connected(slim.flatten(diff_feature), 2, activation_fn=tf.nn.sigmoid)
        _inner_product = tf.reshape(fc_out, [config.BATCH_SIZE, 1, 2])

        result = _inner_product

        fc_left = slim.dropout(slim.fully_connected(slim.flatten(_left), 702, activation_fn=tf.nn.sigmoid))
        fc_right = slim.dropout(slim.fully_connected(slim.flatten(_right), 702, activation_fn=tf.nn.sigmoid))

    restore = tf.train.Saver()
    restore.restore(sess, model_siamese_cnn)
    query_name = []
    test_name = []
    with open(query_name_dir, 'r') as f:
    # with open(train_name_dir, 'r') as f:
        for line in f:
            query_name.append(line[:-1])
    with open(test_name_dir, 'r') as f:
    # with open(train_name_dir, 'r') as f:
        for line in f:
            test_name.append(line[:-1])

    # here is the real test check
    for query in query_name:
        print('probe', query)
        start = time.time()
        name_id = query.split('_')[0]
        # batch_size of the same img.
        left_array = config.BATCH_SIZE * [
            preprocess_input(img_2_array(load_img(query_img_dir + query, target_size=[224, 224])))]
            # preprocess_input(img_2_array(load_img(train_img_dir + query, target_size=[224, 224])))]

        all_real_correct = 0
        count = 0
        right_array = []
        right_name = []

        for test in test_name:

            if count < config.BATCH_SIZE:
                count += 1
                right_name.append(test)

                # comes the accuracy count.
                if test.split('_')[0] == name_id:
                    print('here should correct:', test)
                    all_real_correct += 1

                right_array.append(
                    preprocess_input(img_2_array(load_img(test_img_dir + test, target_size=[224, 224]))))
                    # preprocess_input(img_2_array(load_img(train_img_dir + test, target_size=[224, 224]))))
            else:
                _result, _, __ = sess.run([result, fc_left, fc_right], feed_dict={left: left_array, right: right_array})
                # print(np.argmax(_, 1))
                # print(np.argmax(__, 1))
                count_accuracy_recall(_result)
                count = 0
                right_array = []
                right_name = []

        # if last img not enough to make a batch.
        if right_array and len(right_array) < config.BATCH_SIZE:
            for i in range(config.BATCH_SIZE - len(right_array)):
                right_array.append(list(np.zeros([224, 224, 3])))
            _result, _, __ = sess.run([result, fc_left, fc_right], feed_dict={left: left_array, right: right_array})
            count_accuracy_recall(_result)
        end = time.time()
        print(start-end)
