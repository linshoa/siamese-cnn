import json
from utils.extract_batch import get_id

person_id = get_id(name_select='train')
person_id = sorted(person_id)

id_dict = dict()
data_name_dir = '../data/DukeMTMC/'
train_name_dir = data_name_dir+'bounding_box_train.txt'
save_id_name = open(data_name_dir+'train_id_name.json', 'w')
with open(train_name_dir, 'r') as f:
    for line in f:
        _id = line.split('_')[0]
        if _id in person_id:
            if _id not in id_dict:
                id_dict[_id] = []
            id_dict[_id].append(line[:-1])
# print(id_dict)
json.dump(id_dict, save_id_name)

# save_id_name = open(data_name_dir+'train_id_name.json', 'r')
# id_person = json.load(save_id_name)
# print(id_person)