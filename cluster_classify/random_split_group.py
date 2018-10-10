import numpy as np


def split_id(line_in):
    return line_in.split('_')[0]


# here comes the random select
def random_select(_all_dict):
    all_group = []
    for _id in _all_dict:
        exact_id = _all_dict[_id]
        number = len(exact_id)
        random_group_number = np.random.randint(1, number)
        for group in range(random_group_number):
            # number need to update
            # exact_id need to update
            # todo 1
            if number-random_group_number-group-1 <= 1:
                random_want_select = 1
            else:
                random_want_select = np.random.randint(1, number-random_group_number-group-1)
            print(random_want_select)
            one_group = list(np.random.choice(exact_id, random_want_select, replace=False))
            all_group.append(one_group)
            for i in one_group:
                print(i)
                # try:
                exact_id.remove(i)
                # except Exception:
                #     print(i+'ssssssssss')
            number = len(exact_id)


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

    random_select(_all_id)