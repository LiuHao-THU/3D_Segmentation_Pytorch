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

    def __init__(self, split_images, split_labels, Data_dir,
                                         transform=None, mode='train'):
        super(ScienceDataset, self).__init__()
        start = timer()
        self.transform = transform
        self.mode = mode
        self.Data_dir = Data_dir
        #get image_list
        ids_images = read_names_from_split(split_images)
        ids_labels = read_names_from_split(split_labels)
        #save
        self.ids_images = ids_images
        self.ids_labels = ids_labels

        #print
        print('\ttime = %0.2f min'%((timer() - start) / 60))
        print('\tnum_ids_images = %d'%(len(self.ids_images)))
        print('\tnum_ids_labels = %d'%(len(self.ids_labels)))
        print('')

    def __getitem__(self, index):
        #read image
        id_image = self.ids_images[index]
        image_folder, image_name = id_image.split('/')
        image_dir = os.path.join(self.Data_dir, image_folder, image_name)
        if self.mode in ['train']:
            #read label
            id_label = self.ids_labels[index]
            label_folder, label_name = id_label.split('/')
            label_dir = os.path.join(self.Data_dir, label_folder, label_name)
            #load images and labels with augmentation
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
        return len(self.ids_images)

def chech_read_names_from_split(split_dir):
    txt_file = \
        read_names_from_split(split_dir = split_dir)
    print(txt_file)


if __name__ == '__main__':
    split_dir = '/home/liuh/Documents/3D_pytorch/build/DataReader/split/labels'
    # chech_read_names_from_split(split_dir)