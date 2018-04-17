import os
import torch
import sys
import random
import matplotlib
from utils import *
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '3,2' #'3,2,1,0'
# --------------------------------------------------------------------
PROJECT_PATH = '/home/wanggh/Music/3D_Pytorch'
print('@%s:  ' % PROJECT_PATH)
 
Data_Dir = PROJECT_PATH + '/Data'
out_dir = PROJECT_PATH + '/results'
Split_dir = Data_Dir + '/split'
Check_Pints_dir = out_dir + '/checkpoints'

Mkdir(Check_Pints_dir)

if 1:
    SEED = 35202
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print('\tset random seed')
    print('\t\tSEED=%d' % SEED)
if 1:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print('\tset cuda environment')
    print(
          '\t\ttorch.__version__              =',
          torch.__version__)
    print(
          '\t\ttorch.version.cuda             =',
          torch.version.cuda)
    print(
          '\t\ttorch.backends.cudnn.version() =',
          torch.backends.cudnn.version())
    try:
        print(
            '\t\tos[\'CUDA_VISIBLE_DEVICES\']     =',
            os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =', 'None')
        NUM_CUDA_DEVICES = 1

    print(
          '\t\ttorch.cuda.device_count()      =',
          torch.cuda.device_count())
    print(
        '\t\ttorch.cuda.current_device()    =',
        torch.cuda.current_device())