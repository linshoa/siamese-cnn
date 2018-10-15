import sys
import json
from random import shuffle
from utils.image_precess import *
from utils import config

sys.path.append('./')
sys.path.append('../')
"""
for DukeMTMC example：
0001_c2_f0046302.jpg--> {{'id':0001},{location:c2},{time:0046302},}
"""

name_dir = config.DukeMTMC_name_dir
DukeMTMC_directory = config.DukeMTMC_img_dir
diff_set = {'train': 'bounding_box_train.txt', 'test': 'bounding_box_test.txt', 'query': 'query.txt'}


def get_id(name_select):
    """
    get the set of person id
    :param name_select: whether select train, test or query
    :return: return the no-repeat person id
    """
    """save as the json."""
    person_set_id = []
    name_select = diff_set[name_select]
    with open(name_dir + name_select, 'r') as f:
        for line in f:
            person_set_id.append(line.split('_')[0])
        # set 无序不重复集
        person_set_id = list(set(person_set_id))
    with open('./person_set_id.json', 'w') as w:
        json.dump(person_set_id, w)
    # with open('./utils/person_set_id.json', 'r') as r:
    #     person_set_id = json.load(r)
    return person_set_id


def get_identity(_person_id):
    """will change every time"""
    # identity = [0. for i in range(7140)]
    identity = np.zeros([702])
    index = set_id.index(_person_id)
    identity[int(index)-1] = float(1)
    return identity


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


def get_id_spatio_temporal(line):
    """
    get the id location time of the img_name
    :param line: img_name
    :return: [person_id, location, _time]
    """
    data = line[:-4].split('_')
    person_id, location, _time = data[0], data[1][1: ], data[2][1:]
    person_id = get_identity(person_id)
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
        # positive
        label = [1, 0]
    else:
        left_name = np.random.choice(candidate, 1)[0]
        set_id_copy.remove(person_id)
        person_id_another = np.random.choice(set_id_copy, 1)[0]
        candidate_another = id_dict[person_id_another]
        right_name = np.random.choice(candidate_another, 1)[0]
        # negative
        label = [0, 1]
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
        if person_id.index(i) < len(person_id)//4:
            # positive
            left, right, label = get_exact_id_pair(i, positive=True)
        else:
            # negative
            left, right, label = get_exact_id_pair(i, positive=False)
        left_imgs.append(left)
        right_imgs.append(right)
        labels.append(label)

    # here comes the shuffle
    shuffle_index = np.arange(len(labels))
    shuffle(shuffle_index)
    shuffle_left_imgs = []
    shuffle_right_imgs = []
    shuffle_labels = []
    left_info = []
    right_info = []
    left_id = []
    right_id = []
    
    for index in shuffle_index:
        shuffle_left_imgs.append(left_imgs[index])
        shuffle_right_imgs.append(right_imgs[index])
        shuffle_labels.append(labels[index])
    # labels should convert to row data like (2,1)
    # # in the networks labels should convert it to float32.
    # print(np.asarray(shuffle_labels, dtype='float32')[:, np.newaxis].shape)
    shuffle_labels = list(np.asarray(shuffle_labels, dtype='float32')[:, np.newaxis])
    for left_name in shuffle_left_imgs:
        left_info.append(get_id_spatio_temporal(left_name))
    for right_name in shuffle_right_imgs:
        right_info.append(get_id_spatio_temporal(right_name))
    return shuffle_left_imgs, shuffle_right_imgs, shuffle_labels, left_info, right_info


def precess_to_array(left_imgs_name, right_imgs_name, target_size, name_select):
    data_dir = DukeMTMC_directory+diff_set[name_select][:-4]
    left = list()
    right = list()
    for l in left_imgs_name:
        left.append(preprocess_input(img_2_array(load_img(data_dir+'/'+l, target_size=target_size))))
    for r in right_imgs_name:
        right.append(preprocess_input(img_2_array(load_img(data_dir+'/'+r, target_size=target_size))))
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
        # todo the format of imgs_array is wrong ! while labels not sure!!
        left_imgs_name, right_imgs_name, labels, info_left, info_right = get_pair(_ids, start, end)
        left_imgs_array, right_imgs_array = precess_to_array(left_imgs_name, right_imgs_name, target_size, name_select)
        return left_imgs_array, right_imgs_array, labels, info_left, info_right, end


# load just once!!!
set_id = get_id('train')

# just load once !!
id_dict = get_id_corresponding_name()

if __name__ == '__main__':
    print(get_identity('0001'))