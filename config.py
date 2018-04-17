import os
import torch
config = {}
config['train_batch_size'] = 1
config['valid_batch_size'] = 1
config['test_batch_size'] = 1
config['epochs'] = 200
config['lr'] = 0.01
config['momentum'] = 0.5
config['no-cuda'] = False
config['seed'] = 1
config['log_interval'] = 1
config['cuda'] = torch.cuda.is_available() #args.no_cuda 
config['label_split_dir'] = '/home/liuh/Documents/3D_pytorch/build/DataReader/split/labels'
config['image_split_dir'] = '/home/liuh/Documents/3D_pytorch/build/DataReader/split/images'
config['crop_size'] = [48, 96, 96]
config['num_workers'] = 4
config['pin_memory'] = True
config['drop_last'] = True
config['save_epoch_num'] = 2