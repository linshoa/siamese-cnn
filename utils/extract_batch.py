import json
import numpy as np
from random import shuffle
from utils.image_precess import *
import config

"""
for DukeMTMC example：
0001_c2_f0046302.jpg--> {{'id':0001},{location:c2},{time:0046302},}
"""

name_dir = '../data/DukeMTMC/'
DukeMTMC_directory = config.DukeMTMC_img_dir
diff_set = {'train': 'bounding_box_train.txt', 'test': 'bounding_box_test.txt', 'query': 'query.txt'}


def get_id(name_select):
    """
    get the set of person id
    :param name_select: whether select train, test or query
    :return: return the no-repeat person id
    """
    person_set_id = []
    name_select = diff_set[name_select]
    with open(name_dir + name_select, 'r') as f:
        for line in f:
            person_set_id.append(line.split('_')[0])
        # set 无序不重复集
        person_set_id = list(set(person_set_id))
    return person_set_id


# load just once!!!
set_id = get_id('train')


def get_id_corresponding_name():
    """
    reload the json.
    the json have save the id and its corrsponding img.
    like : '0178': ['0178_c1_f0086107.jpg', '0178_c1_f0086227.jpg', '0178_c1_f0086347.jpg'...]
    :return: dict
    """
    save_id_name = open(name_dir + 'train_id_name.json', 'r')
    id_person = json.load(save_id_name)
    # print(id_person)
    return id_person


# just load once !!
id_dict = get_id_corresponding_name()


def get_id_spatio_temporal(line):
    """
    get the id location time of the img_name
    :param line: img_name
    :return: [person_id, location, _time]
    """
    data = line[:-4].split('_')
    person_id, location, _time = data[0], data[1], data[2][1:]
    return [person_id, location, _time]


# another reference : https://blog.csdn.net/appleml/article/details/57413615
# 另外需要注意的是，前三种方式只是所有语料遍历一次，而最后一种方法是，所有语料遍历了num_epochs次

# reference: https://github.com/digitalbrain79/person-reid
# here comes a idea, a image in the batch may be the one the that have appear in the last batch


def get_exact_id_pair(person_id, positive):
    set_id_copy = set_id.copy()
    candidate = id_dict[person_id]
    if positive:
        left_name, right_name = list(np.random.choice(candidate, 2))
        label = 1
    else:
        left_name = np.random.choice(candidate, 1)[0]
        set_id_copy.remove(person_id)
        person_id_another = np.random.choice(set_id_copy, 1)[0]
        candidate_another = id_dict[person_id_another]
        right_name = np.random.choice(candidate_another, 1)[0]
        label = 0
    return left_name, right_name, label


def get_pair(_ids, start, end):
    left_imgs = list()
    right_imgs = list()
    labels = list()
    if start < end:
        person_id = _ids[start:end]
    else:
        person_id = _ids[start:] + _ids[:end]

    # split into positive and negative
    for i in person_id:
        if person_id.index(i) < len(person_id)//2:
            # positive
            left, right, label = get_exact_id_pair(i, positive=True)
        else:
            # negative
            left, right, label = get_exact_id_pair(i, positive=False)
        left_imgs.append(left)
        right_imgs.append(right)
        labels.append(label)

    # here comes the shuffle
    # since
    shuffle_index = np.arange(len(labels))
    shuffle(shuffle_index)
    shuffle_left_imgs = []
    shuffle_right_imgs = []
    shuffle_labels = []
    for index in shuffle_index:
        shuffle_left_imgs.append(left_imgs[index])
        shuffle_right_imgs.append(right_imgs[index])
        shuffle_labels.append(labels[index])
    # labels should convert to row data like (2,1)
    # in the networks labels should convert it to float32.
    shuffle_labels = np.asarray(shuffle_labels)[:, np.newaxis]
    return shuffle_left_imgs, shuffle_right_imgs, shuffle_labels


def precess_to_array(left_imgs_name, right_imgs_name, target_size, name_select):
    data_dir = DukeMTMC_directory+diff_set[name_select][:-4]
    left = list()
    right = list()
    for l in left_imgs_name:
        left.append(img_2_array(load_img(data_dir+'/'+l, target_size)))
    for r in right_imgs_name:
        right.append(img_2_array(load_img(data_dir+'/'+r, target_size)))
    return left, right


def next_batch(batch_size, target_size, is_train, start):
    """
    get the next batch
    :param batch_size:
    :param target_size: []
    :param is_train: bool
    :return: left_imgs_array, right_imgs_array, labels, end
    """
    if is_train:
        name_select = 'train'
        _ids = get_id(name_select)
        # you'd better just get the name of the img here
        # remember the rank you get positive and negative should not only
        # make sure the ratio is 1:1 but also shuffle the order.
        end = (start+batch_size) % len(_ids)
        # positive pair add in sequence of the set!!(so just take care the odd),
        # while negative just randomly select pairs
        # take care start > end !!!
        left_imgs_name, right_imgs_name, labels = get_pair(_ids, start, end)
        left_imgs_array, right_imgs_array = precess_to_array(left_imgs_name, right_imgs_name, target_size, name_select)
        return left_imgs_array, right_imgs_array, labels, end


if __name__ == '__main__':
    next_batch(2, [224, 224], True, 702)