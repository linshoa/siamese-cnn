import os

# data name : Martket-1501
# purpose : restore all the info of the images in the text, so we can extract the images from the text conveniently.
file_directory = '/home/ubuntu/media/File/1Various/Person_reid_dataset/Market-1501-v15.09.15/'
# bounding_box_test is for gallery.   19732 images for 750 individuals
# bounding_box_train is for training. 12936 images for 751 individuals
# query is for querying  during test. 3368 query images

# Attention!: Names beginning with "0000" are distractors produced by DPM false detection.
# Names beginning with "-1" are junks that are neither good nor bad DPM bboxes.
# So "0000" will have a negative impact on accuracy, while "-1" will have no impact.
# During testing, we rank all the images in "bounding_box_test".
# Then, the junk images are just neglected; distractor images are not neglected.

train = 'bounding_box_train'
test = 'bounding_box_test'
query = 'query'

usage = [train, test, query]
for item in usage:
    if not os.path.exists('./market-1501/'):
        os.makedirs('./market-1501/')
    output = open('./market-1501/'+item + '.txt', 'w')
    file_name = os.listdir(file_directory+item+'/')
    file_name.sort()
    for name in file_name:
        if name.endswith('.jpg'):
            output.writelines(name+'\r\n')
    output.close()