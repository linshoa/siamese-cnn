from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import utils.config as config
from utils.extract_batch import next_batch

resnet_v1_50_model = './model/resnet_v1_50.ckpt'
model_input = './model/only_visual_result/model_no_dropout_1_3_49600.ckpt'


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def contrastive_loss(_left, _right, _label_input, _left_label, _right_label):
    with tf.name_scope('output'):
        """RANDOM"""
        # here is for random cos
        # first divide the feature into 3 group
        # them random walk, and accept the max feature.
        # left_all = []
        # right_all = []
        # for i in range(4):
        #     left_all.append(_left[:, _left.shape[1]//4*i:_left.shape[1]//4*(i+1), :, :])
        #     right_all.append(_right[:, _right.shape[1]//4*i:_right.shape[1]//4*(i+1), :, :])
        #
        # for i in range(4):
        #     diff_feature_random_walk = list()
        #     for j in range(4):
        #         diff_feature_random_walk.append(tf.matmul(tf.nn.l2_normalize(left_all[i]), tf.nn.l2_normalize(right_all[j]), transpose_a=False, transpose_b=True))
        #
        #     if not i:
        #         feature_out = tf.reduce_max(diff_feature_random_walk, axis=0)
        #     else:
        #         feature_out = tf.concat([feature_out, tf.reduce_max(diff_feature_random_walk, axis=0)], axis=1)
        # del left_all, right_all
        # del diff_feature_random_walk
        # fc_out = slim.fully_connected(slim.flatten(feature_out), 2, activation_fn=None)
        # _inner_product = tf.reshape(fc_out, [config.BATCH_SIZE, 1, 2])

        """cos"""
        _left = slim.flatten(_left)
        _right = slim.flatten(_right)
        _left_l2 = tf.nn.l2_normalize(_left, name='left_l2_norm')
        _right_l2 = tf.nn.l2_normalize(_right, name='right_l2_norm')
        diff_feature = tf.multiply(_left_l2, _right_l2)
        print('shape:' + str(diff_feature.shape))
        fc_out = slim.fully_connected(slim.flatten(diff_feature), 2, activation_fn=tf.nn.sigmoid)
        _inner_product = tf.reshape(fc_out, [config.BATCH_SIZE, 1, 2])
        # 输出 [1,0] or [0, 1]

        fc_left = slim.fully_connected(slim.flatten(_left), 702, activation_fn=tf.nn.sigmoid)
        fc_right = slim.fully_connected(slim.flatten(_right), 702, activation_fn=tf.nn.sigmoid)

    with tf.name_scope('loss'):
        _left_id_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=_left_label, logits=fc_left)
        _right_id_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=_right_label, logits=fc_right)

        _loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=_label_input, logits=_inner_product)

        _loss_out = 2 * tf.reduce_mean(_loss) + tf.reduce_mean(_left_id_loss) + tf.reduce_mean(_right_id_loss)
        # _loss_out = tf.reduce_mean(_left_id_loss) + tf.reduce_mean(_right_id_loss)
    return _loss_out, fc_left, fc_right, _inner_product


if __name__ == '__main__':
    with tf.name_scope('input'):
        left = tf.placeholder(tf.float32, [None, 224, 224, 3], name='left')
        right = tf.placeholder(tf.float32, [None, 224, 224, 3], name='right')

    with tf.name_scope('label'):
        label_input = tf.placeholder(tf.float32, [None, 1, 2], name='label')
        identity_left_label = tf.placeholder(tf.float32, [None, 702], name='identity_left_label')
        identity_right_label = tf.placeholder(tf.float32, [None, 702], name='identity_right_label')

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        # for the paper of vehicle, global_pool is True
        left_feature, _ = nets.resnet_v1.resnet_v1_50(left, is_training=True, global_pool=False, reuse=False)
        right_feature, __ = nets.resnet_v1.resnet_v1_50(right, is_training=True, global_pool=False, reuse=True)
        print(right_feature.shape)
        # if global_pool True, which means global average
        # shape is (batch, 1, 1, 2048)
        # while if False
        # shape is (batch, x, x, 2048)

    with tf.Session() as sess:
        lr = 1e-5
        # restore the graph, so we should not define any other graph before that.
        restore = tf.train.Saver()
        restore.restore(sess, resnet_v1_50_model)
        # after that, can be other networks
        loss, left_identity, right_identity, _cos = contrastive_loss(left_feature, right_feature, label_input, identity_left_label, identity_right_label)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
        initialize_uninitialized(sess)
        # tf.global_variables_initializer() wrong!!!!!
        # todo some problem. now want fine-tune but find the acc is 0.0
        # TODO so maybe the save own this problem.no resnet.

        # define something to save
        saver = tf.train.Saver(max_to_keep=2)

        # setup for tensorboard
        # tf.summary.scalar('step', global_step)
        # tf.summary.scalar('loss', loss)
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter('./model/only_visual_result/', sess.graph)

        # training epoch
        next_start = 0
        for i in range(config.EPOCH):

            # todo tf.data.Dataset.from_generator()

            left_array, right_array, label_float32, _info_left, _info_right, next_start = next_batch(config.BATCH_SIZE, [224, 224], True, next_start)
            # print(np.array(_info_left).transpose())
            # print(np.array(_info_left).transpose()[0])
            _id_left_label = np.array(list(np.array(_info_left).transpose()[0]))
            # print(_id_left_label.shape)
            # print(_id_left_label)
            _id_right_label = np.array(list(np.array(_info_right).transpose()[0]))

            _train, sum_loss, _left_identity, _right_identity, __cos = sess.run([train_step, loss, left_identity, right_identity, _cos], feed_dict={left: left_array, right: right_array, identity_left_label: _id_left_label, identity_right_label: _id_right_label, label_input: label_float32})

            # writer.add_summary(summary_str, i)
            if i % config.SAVE_ITER == 0 and i != 0:

                saver.save(sess, './model/only_visual_result/model_no_dropout_1_3_%d.ckpt' % i)
                if lr > 1e-11:
                    lr /= 2
            print(i, sum_loss)
            cos_acc = np.mean(np.argmax(__cos, 2) == np.argmax(label_float32, 2))
            left_accuracy = np.mean(np.argmax(_left_identity, 1) == np.argmax(_id_left_label, 1))
            right_accuracy = np.mean(np.argmax(_right_identity, 1) == np.argmax(_id_right_label, 1))
            print('left_accuracy:', left_accuracy, 'right_accuracy:', right_accuracy, 'all_accuracy:', np.mean([left_accuracy, right_accuracy]))
            print('cos_acc', cos_acc)
