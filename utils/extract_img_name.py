import os

# data name : Martket-1501
# purpose : restore all the info of the images in the text, so we can extract the images from the text conveniently.
market_directory = '/home/ubuntu/media/File/1Various/Person_reid_dataset/Market-1501-v15.09.15/'
# bounding_box_test is for gallery.   19732 images for 750 individuals
# bounding_box_train is for training. 12936 images for 751 individuals
# query is for querying  during test. 3368 query images
#
# Attention!: Names beginning with "0000" are distractors produced by DPM false detection.
# Names beginning with "-1" are junks that are neither good nor bad DPM bboxes.
# So "0000" will have a negative impact on accuracy, while "-1" will have no impact.
# During testing, we rank all the images in "bounding_box_test".
# Then, the junk images are just neglected; distractor images are not neglected.


# data name : DukeMTMC-reID
DukeMTMC_directory = '/home/ubuntu/media/File/1Various/Person_reid_dataset/DukeMTMC-reID/'
# TOTALLY, 36,411 bounding boxes with IDs. There are 1,404 identities appearing in more than two cameras
# and 408 identities (distractor ID) who appear in only one camera. We randomly select 702 IDs as the training set
# and the remaining 702 IDs as the testing set.
# In the testing set, we pick one query image for each ID in each camera and put the remaining images in the gallery
# As a result, we get 16,522 training images of 702 identities,
# 2,228 query images of the other 702 identities
# 17,661 gallery images

_train = 'bounding_box_train'
_test = 'bounding_box_test'
_query = 'query'

dataset = {'market-1501': market_directory, 'DukeMTMC': DukeMTMC_directory}
usage = [_train, _test, _query]
for set in dataset:
    for item in usage:
        if not os.path.exists('../data/'+set+'/'):
            os.makedirs('../data/'+set+'/')
        output = open('../data/'+set+'/'+item + '.txt', 'w')
        file_name = os.listdir(dataset[set]+item+'/')
        file_name.sort()
        for name in file_name:
            if name.endswith('.jpg'):
                output.writelines(name+'\r\n')
        output.close()
