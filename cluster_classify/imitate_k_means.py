import pandas as pd
import numpy as np
from cluster_classify.img_to_feature import get_feature


def get_distance(vector_a, vector_b):
    """
    :param vector_a: [H, W, C]
    :param vector_b: [H, W, C]
    :return: only one number
    """
    a = np.square(pow(vector_a, 2))
    b = np.square(pow(vector_b, 2))
    return sum(np.matmul(np.transpose(a), b))


def get_centers(batch_data):
    """
    :param batch_data: [B, H, W, C] (n,8,3,2048)
    :return: [H, W, C] --> (8, 3, 2018)
    """
    h_mean = np.mean(batch_data[:, 8, :, :], 0)
    w_mean = np.mean(batch_data[:, :, 3, :], 0)
    c_mean = np.mean(batch_data[:, :, :, 2018], 0)
    return np.array([h_mean, w_mean, c_mean])


def k_means(group_label):
    """
    :param group_label: big list [[group1], [group2]]
    :return: group_label
    """
    old_distance = 0
    new_distance = 1
    print(group_label)
    while new_distance != old_distance:
        old_distance = new_distance
        centers = []
        center_to_center_distance = []
        for group_number, _ in enumerate(group_label):
            batch_data = []
            for img_name in group_label[group_number]:
                batch_data.append(get_feature(img_name))
            batch_center = get_centers(batch_data)
            batch_distance = 0
            for i in batch_data:
                batch_distance += get_distance(i, batch_center)
            new_distance += batch_distance / len(batch_data)
            centers.append(get_centers(batch_data))
        for index, centers_one in enumerate(centers):
            for centers_another in centers[index+1:]:
                center_to_center_distance.append(get_distance(centers_one, centers_another))
        center_index = center_to_center_distance.index(min(center_to_center_distance))
        # i is the centers_one, j is the centers_another
        i = 1
        center_index += 1
        while center_index > len(centers)-i:
            center_index -= len(centers)-i
            i += 1
        j = center_index + i - 1
        i -= 1
        # centers[i] and centers[j] should try cluster equal to the group_label group togother.
        for to_group in group_label[j]:
            group_label[i].append(to_group)
        group_label.remove(group_label[j])
    print(group_label)
    # here should cluster different points.
    # but remember the cost function is the distances between all points to the center points


def load_group_data():
    file_dir = './random_group.json'
    df = pd.read_json(file_dir)
    big_list_group = []
    for group_img, index in df.values:
        # transfer into big list !!!
        big_list_group.append(group_img)
    return big_list_group


if __name__ == '__main__':
    group_label = load_group_data()
    k_means(group_label)