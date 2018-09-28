import os
import tensorflow as tf

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# 我们可以通过设置 CUDA_DEVICE_ORDER = PCI_BUS_ID 来要求运行时
# 设备查询按照 PCI_BUS_ID 的顺序索引，从而使得 设备ID=物理ID 保证CUDA应用按期望使用指定设备。
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

