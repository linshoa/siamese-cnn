import utils.cuda_util as cuda
import utils.config as config
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from utils.image_precess import *
import numpy as np
import time

model_siamese_cnn = './model/result/model_4950.ckpt'
query_img_dir = config.DukeMTMC_img_dir+'query/'
test_img_dir = config.DukeMTMC_img_dir+'bounding_box_test/'
query_name_dir = config.DukeMTMC_name_dir+'query.txt'
test_name_dir = config.DukeMTMC_name_dir+'bounding_box_test.txt'

with tf.Session(config=cuda.config) as sess:
    with tf.name_scope('input'):
        left = tf.placeholder(tf.float32, [None, 240, 80, 3], name='left')
        right = tf.placeholder(tf.float32, [None, 240, 80, 3], name='right')

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        # for the paper of vehicle, global_pool is True
        left_feature, _ = nets.resnet_v1.resnet_v1_50(left, is_training=True, global_pool=False, reuse=False)
        right_feature, __ = nets.resnet_v1.resnet_v1_50(right, is_training=True, global_pool=False, reuse=True)

    with tf.name_scope('output'):
        # the transpose here really memory consume!!!
        # inner_product = tf.matmul(_left, tf.matrix_transpose(_right))
        #todo drpout
        diff_feature = tf.square(left_feature - right_feature)
        fc_out = slim.fully_connected(slim.flatten(diff_feature), 2, activation_fn=None)
        _inner_product = tf.reshape(fc_out, [config.BATCH_SIZE, 1, 2])
        result = slim.softmax(_inner_product)
        # _inner_product = tf.reshape(fc_out, [config.BATCH_SIZE, 1])
        # similarity = slim.nn.sigmoid(inner_product)

    restore = tf.train.Saver()
    restore.restore(sess, model_siamese_cnn)
    with open(query_name_dir, 'r') as f:
        for line in f:
            mAp = []
            final_recall = []
            all_real_correct = 0
            find_real_correct = 0
            judge_correct = 0

            name = line[:-1]
            name_id = name.split('_')[0]
            left_array = []
            for i in range(config.BATCH_SIZE):
                # batch_size of the same img.
                left_array.append(preprocess_input(img_2_array(load_img(query_img_dir+name, target_size=[240, 80]))))
            with open(test_name_dir, 'r') as test:
                count = 0
                right_array = []
                right_name = []
                for test_line in test:
                    if count < config.BATCH_SIZE:
                        count += 1
                        name_test = test_line[:-1]
                        right_name.append(name_test)
                        if name_test.split('_')[0] == name_id:
                            all_real_correct += 1
                        right_array.append(preprocess_input(img_2_array(load_img(test_img_dir + name_test, target_size=[240, 80]))))

                    else:
                        # print('probe:' + name)
                        _result = sess.run([result], feed_dict={left: left_array, right: right_array})
                        # for i in _score:
                        #     print(i[0])
                        # default: ascend todo descend
                        i = 0
                        for batch_i in _result[0]:
                            # print(batch_i[0])
                            if np.argmax(batch_i[0]) == 0:
                                judge_correct += 1
                                if right_name[i].split('_')[0] == name_id:
                                    find_real_correct += 1
                                # print(right_name[i])
                                # print(True)
                                # time.sleep(2)

                            # elif np.argmax(batch_i[0]) == 1:
                            #     print(False)
                            i += 1

                        count = 0
                        right_array = []
                        right_name = []
            if right_array and len(right_array) < config.BATCH_SIZE:
                # print(right_array)
                # print(len(right_array))
                for ___ in range(config.BATCH_SIZE-len(right_array)):
                    right_name.append(name)
                    right_array.append(preprocess_input(img_2_array(load_img(query_img_dir+name, target_size=[240, 80]))))
                print('probe:' + name)
                _result = sess.run([result], feed_dict={left: left_array, right: right_array})
                # for i in _score:
                #     print(i[0])
                # default: ascend todo descend
                i = 0
                for batch_i in _result[0]:
                    # print(batch_i[0])
                    if np.argmax(batch_i[0]) == 0:
                        if right_name[i] != name:
                            judge_correct += 1
                            if right_name[i].split('_')[0] == name_id:
                                find_real_correct += 1
                            # print(right_name[i])
                            # print(True)
                    i += 1

            ap = find_real_correct / judge_correct
            print(ap)
            recall = find_real_correct / all_real_correct
            print(recall)
            mAp.append(ap)
            final_recall.append(recall)
    print(sum(mAp) / len(mAp))
    print(sum(final_recall) / len(final_recall))

    # take a query
    # for every img in the test, compute the distance
    # rank the distance and save the distance and name
