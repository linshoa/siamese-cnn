import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import utils.config as config
import utils.cuda_util as cuda
from utils.extract_batch import next_batch

resnet_v1_50_model = './model/resnet_v1_50.ckpt'


def contrastive_loss(_left, _right, _label_input):
    with tf.name_scope('output'):
        inner_product = tf.matmul(_left, tf.matrix_transpose(_right))
        _inner_product = tf.reshape(inner_product, [config.BATCH_SIZE, 1])
        # similarity = slim.nn.sigmoid(inner_product)

    with tf.name_scope('loss'):
        _loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=_label_input, logits=_inner_product)
        __loss_out = tf.reduce_sum(_loss)
    return __loss_out


# global_step = tf.Variable(0, trainable=False)
# train_step = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)


if __name__ == '__main__':
    with tf.name_scope('input'):
        left = tf.placeholder(tf.float32, [None, 240, 80, 3], name='left')
        right = tf.placeholder(tf.float32, [None, 240, 80, 3], name='right')

    with tf.name_scope('label'):
        label_input = tf.placeholder(tf.float32, [None, 1], name='label')

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        # todo is_training True
        # for the paper of vehicle, global_pool is True
        left_feature, _ = nets.resnet_v1.resnet_v1_50(left, is_training=False, global_pool=True, reuse=False)
        right_feature, __ = nets.resnet_v1.resnet_v1_50(right, is_training=False, global_pool=True, reuse=True)
        # if global_pool True, which means global average
        # shape is (batch, 1, 1, 2048)
        # while if False
        # shape is (batch, x, x, 2048)
        print(left_feature.shape)
        print(right_feature.shape)
    with tf.Session(config=cuda.config) as sess:
        # sess.run(tf.global_variables_initializer())
        restore = tf.train.Saver()
        restore.restore(sess, resnet_v1_50_model)
        left_array, right_array, label_int, next_start = next_batch(config.BATCH_SIZE, [240, 80], True, 0)

        _left_out, _right_out = sess.run([left_feature, right_feature], feed_dict={left:left_array, right:right_array, label_input:label_int})
        print(_left_out.shape)
        print(_right_out.shape)
        loss = contrastive_loss(_left_out, _right_out, label_input)
        print(sess.run(loss))