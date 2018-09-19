import cv2
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets




class Siamese_cnn():
    resnet_v1_model = './model/resnet_v1_50.ckpt'
    image_dir = './data/market-1501/'
    def __init__(self):
        pass

    def batch_next(self):
        pass

    def batch_input_data(self):
        name, label = Siamese_cnn.batch_next()
        image = []
        for i in range(len(name)):
            image.append(cv2.imread(self.image_dir+name[i]).astype(np.float32))
            # todo 400????
            image[i] = cv2.resize(image[i], (400, 224), interpolation=cv2.INTER_CUBIC)
            image[i] = image[i].reshape(1, 400, 224, 3)
        image_all = np.concatenate((image[::]), 0)
        #todo
        label_all = 'xxx'
        return image_all, label_all

    def model_structure(self, input):
        image_all, label_all = self.batch_input_data()
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            nets.resnet_v1.resnet_v1_50(image_all, is_training=False, global_pool=False)