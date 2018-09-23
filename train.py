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
        # the transpose here really memory consume!!!
        # inner_product = tf.matmul(_left, tf.matrix_transpose(_right))
        #todo drpout
        diff_feature = tf.square(_left-_right)
        # fc_out = slim.stack(slim.flatten(diff_feature), slim.fully_connected(activation_fn=slim.nn.tanh), [1024, 256, 1], scope='fc')
        # fc = slim.fully_connected(slim.flatten(diff_feature), 1024, activation_fn=slim.nn.tanh)
        # fc = slim.fully_connected(fc, 256, activation_fn=slim.nn.relu)
        fc_out = slim.fully_connected(slim.flatten(diff_feature), 1, activation_fn=None)
        _inner_product = tf.reshape(fc_out, [config.BATCH_SIZE, 1])
        # similarity = slim.nn.sigmoid(inner_product)

    with tf.name_scope('loss'):
        _loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=_label_input, logits=_inner_product)
        __loss_out = tf.reduce_sum(_loss)
    return __loss_out, _loss


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
        left_feature, _ = nets.resnet_v1.resnet_v1_50(left, is_training=True, global_pool=False, reuse=False)
        right_feature, __ = nets.resnet_v1.resnet_v1_50(right, is_training=True, global_pool=False, reuse=True)
        # if global_pool True, which means global average
        # shape is (batch, 1, 1, 2048)
        # while if False
        # shape is (batch, x, x, 2048)

    with tf.Session() as sess:
        lr = 1e-3
        # restore the graph, so we should not define any other graph before that.
        sess.run(tf.global_variables_initializer())
        restore = tf.train.Saver()
        restore.restore(sess, resnet_v1_50_model)
        # after that, can be other networks
        global_step = tf.Variable(0, trainable=False)
        loss, intermediate_loss = contrastive_loss(left_feature, right_feature, label_input)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
        sess.run(tf.global_variables_initializer())

        # define something to save
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

        # setup for tensorboard
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./model/result/', sess.graph)

        # training epoch
        next_start = 0
        for i in range(config.EPOCH):
            left_array, right_array, label_float32, next_start = next_batch(config.BATCH_SIZE, [240, 80], True, next_start)
            _, sum_loss, not_sum_loss, summary_str = sess.run([train_step, loss,  intermediate_loss, merged], feed_dict={left: left_array, right: right_array, label_input: label_float32})

            writer.add_summary(summary_str, i)
            if i % config.SAVE_ITER == 0 and i != 0:
                saver.save(sess, './model/result/model_%d.ckpt' % i)
                # if config.LEARNING_RATE >= 1e-4:
                # lr /= 10
            print(i, sum_loss)
            # print(_left_out.shape)
            # print(_right_out.shape)
            # problem like Attempting to use uninitialized value fully_connected/biases
            # solved: the position initializer should place after the graph.