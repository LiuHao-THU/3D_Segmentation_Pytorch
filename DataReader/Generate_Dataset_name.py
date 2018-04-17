import os
import sys
sys.path.append("..")
from utils import Mkdir

def last_9chars(x):
    return(x[-9:-4])

def ImageLits2txt(image_list, txt_file_name, folder_name):
	txt_file_name = './split/' + txt_file_name
	txt_file = open(txt_file_name, 'w')
	for img_name in image_list:
		txt_file.write(os.path.join(folder_name, img_name))
		txt_file.write('\n')
	txt_file.close()

def Make_Dataset_txt(Dataset_dir):
	image_dir = Dataset_dir + '/images'
	label_dir = Dataset_dir + '/labels'
	#save image txt file
	image_list = os.listdir(image_dir)
	image_list = sorted(image_list, key = last_9chars)
	label_list = os.listdir(label_dir)
	label_list = sorted(label_list, key = last_9chars)
	#image_list2txt
	ImageLits2txt(image_list = image_list, txt_file_name = 'images', 
		folder_name = 'images')
	ImageLits2txt(image_list = label_list, txt_file_name = 'labels',
		folder_name = 'labels')
	print('succuss!')
     	
def check_txt_file(split, Data_dir):
    with open(split) as f:
        content = f.readlines()
    for line in content:
        print(line)

if __name__ == '__main__':
	# Dataset_dir = '/home/liuh/Desktop/datasets/data'
	# txt_file_dir = os.path.join('./', 'split', 'Data_set')
	# Dataset_dir, image_txt_name, label_txt_name
	# Make_Dataset_txt(Dataset_dir = Dataset_dir)
	
	# check_txt_file(split)