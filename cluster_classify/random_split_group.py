import numpy as np
import pandas as pd


def split_id(line_in):
    return line_in.split('_')[0]


# here comes the random select
def random_select(_all_dict):
    all_group = []
    for _id in _all_dict:
        exact_id = _all_dict[_id]
        number = len(exact_id)
        random_group_number = np.random.randint(2, number)
        for group_index in range(random_group_number):
            # number need to update
            # exact_id need to update

            # the group left need at least one imgs. so number-random_group_number+group_index+1 is the upper bound.
            # print(number-random_group_number+group_index+1)
            if number-random_group_number+group_index+1 > 2:
                random_want_select = np.random.randint(2, number-random_group_number+group_index+1)
            else:
                random_want_select = np.random.randint(1, 2)
            # print(random_want_select)
            one_group = list(np.random.choice(exact_id, random_want_select, replace=False))
            all_group.append(one_group)
            for i in one_group:
                # print(i)
                exact_id.remove(i)
            number = len(exact_id)
            all_group.append(one_group)
    return all_group


if __name__ == '__main__':
    train_dir = '../data/DukeMTMC/bounding_box_train.txt'
    _all_id = {}
    with open(train_dir, 'r') as f:
        for line in f:
            line_id = split_id(line)
            if line_id not in _all_id:
                _all_id[line_id] = []
                _all_id[line_id].append(line[:-1])
            else:
                _all_id[line_id].append(line[:-1])

    first_group = random_select(_all_id)
    _dict_all = {}
    _dict_all['index'] = list()
    _dict_all['group_img'] = list()
    for index, one_group in enumerate(first_group):
        _dict_all['index'].append(index)
        _dict_all['group_img'].append(one_group)
    df = pd.DataFrame(_dict_all)

    df.to_json('./random_group.json')

    df = pd.read_json('./random_group.json')
    df.sort_values(by='index')
    for index, one_group in df.values:
        print(index, one_group)