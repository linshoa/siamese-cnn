import json
import numpy as np


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
    :param batch_data: [B, H, W, C]
    :return: [H, W, C]
    """
    h_mean = np.mean(batch_data, 1)
    w_mean = np.mean(batch_data, 2)
    c_mean = np.mean(batch_data, 3)
    return np.array([h_mean, w_mean, c_mean])


def k_means(feature_data, group_label):
    """
    :param feature_data: dict {'img_name': feature}
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
        for group_number in group_label:
            batch_data = []
            for img_name in group_label[group_number]:
                batch_data.append(feature_data[img_name])
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
    with open(file_dir, 'r') as f:
        random_group = json.load(f)
    # transfer into big list !!!
    big_list_group = []
    for i in random_group:
        big_list_group.append(random_group[i])
    return big_list_group


def load_feature_data():
    file_dir = './img_feature.json'
    with open(file_dir, 'r') as f:


if __name__ == '__main__':
    group_label = load_group_data()