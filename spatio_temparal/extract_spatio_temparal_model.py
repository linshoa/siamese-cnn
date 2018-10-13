import json
import numpy as np
import matplotlib.pyplot as plt


def save_mac_all():
    file_train = '../data/DukeMTMC/bounding_box_train.txt'
    data_all = {}
    with open(file_train, 'r') as file_read:
        for line in file_read:
            # example : 5388_c8_f0005157.jpg
            # idea : for one camera {id : time}
            print(line)
            line_data = line.split('_')
            _id = line_data[0]
            _camera = line_data[1]
            _time = line_data[2][1:-5]
            if _camera not in data_all:
                data_all[_camera] = {}

            if _id not in data_all[_camera]:
                data_all[_camera][_id] = []

            if _time not in data_all[_camera][_id]:
                data_all[_camera][_id].append(_time)

    with open('./Duke_train.json', 'w') as w:
        json.dump(data_all, w)


def load_mac_all():
    file_input = './Duke_train.json'
    with open(file_input, 'r') as o:
        data_all = json.load(o)
    camera_probe = 'c3'
    camera_duration = {}
    camera_probe = data_all[camera_probe]
    for probe_id in camera_probe:
        for probe_time in camera_probe[probe_id]:
            for camera_query in data_all:
                if camera_query != camera_probe:
                    for query_id in data_all[camera_query]:
                        if probe_id == query_id:
                            for query_time in data_all[camera_query][query_id]:
                                duration = int(probe_time) - int(query_time)
                                if duration not in camera_duration:
                                    camera_duration[duration] = 1
                                else:
                                    camera_duration[duration] += 1

    with open('./c3.json', 'w') as w:
        json.dump(camera_duration, w)


def plot():
    with open('./c3.json', 'r') as w:
        camera_duration = json.load(w)
    data_sort = np.array(sorted(camera_duration.items(), key=lambda a: int(a[0])), int)
    plt.xlim(-2000, 2000)
    plt.ylim(0, 1000)
    plt.plot(data_sort[:, 0][::], data_sort[:, 1][::], '-', lw=1)

    plt.show()


if __name__ == '__main__':
    load_mac_all()
    plot()
