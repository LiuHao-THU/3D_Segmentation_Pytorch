import os
from time import time as timer
from skimage import io
import numpy as np
def read_names_from_split(split_dir):
    """read split txt file from source"""
    file_dir_list = []
    with open(split_dir) as f:
        content = f.readlines()
    for line in content:
        #save the data as list
        file_dir_list.append(line[0:-1])
    return file_dir_list

class ScienceDataset():

    def __init__(self, split, Data_dir, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()
        start = timer()
        self.transform = transform
        self.mode = mode
        self.Data_dir = Data_dir
        split_dir = os.path.join(Data_dir, 'split', split)
        # get image_list
        print(split_dir)
        print('this si split_dir')
        ids = read_names_from_split(split_dir)
        # save
        self.ids = ids
        print(self.ids)
        # print
        print('\ttime = %0.2f min' % ((timer() - start) / 60))
        print('\tnum_ids_images = %d' % (len(self.ids)))
        print('')

    def __getitem__(self, index):
        # read image
        id_image = self.ids[index]
        image_folder, image_name = id_image.split('/')
        image_dir = os.path.join(self.Data_dir,
                                 'images',
                                 'TH_' + image_name)
        if self.mode in ['train']:
            # read label
            label_dir = os.path.join(self.Data_dir,
                                     'labels',
                                     image_name)
            # load images and labels with augmentation
            image = io.imread(image_dir).astype(np.float32)
            label = io.imread(label_dir).astype(np.int32)
            if self.transform is not None:
                return self.transform(image, label, index)
            else:
                input = image
                return input, label, index

        if self.mode in ['test']:
            #load images and labels no need transform 
            if self.transform is not None:
                image = io.imread(image_dir).astype(np.float32)
                return self.transform(image, index)
            else:
                return image, index

    def __len__(self):
        return len(self.ids)


def chech_read_names_from_split(split_dir):
    txt_file = \
        read_names_from_split(split_dir = split_dir)
    print(txt_file)


# if __name__ == '__main__':
    # split_dir = '/home/liuh/Documents/3D_pytorch/build/DataReader/split/labels'
    # chech_read_names_from_split(split_dir)