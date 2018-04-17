# -*- coding: utf-8 -*-
# @Author: antigen
# @Date:   2018-04-12 22:39:01
# @Last Modified by:   antigen
# @Last Modified time: 2018-04-15 23:53:05
import torch
from common import *
import torch.nn as nn
from config import config
import torch.optim as optim
import torch.utils as utils
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
from DataReader.sampler import *
import torchvision.utils as v_utils
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from time import time as timer
from losses import DICELossMultiClass
# from Model.model import Net
# from Model.Unet_Zoo import UNet
from Model.UnetGenerator_3d import *
from DataReader.reader import ScienceDataset
from DataReader.transform import Flip_image, Random_crop
# define log
log = Logger()

start = timer()
#define loss function
criterion = DICELossMultiClass()

def loss_function(output,label):
    batch_size,channel,x,y,z = output.size()
    total_loss = 0
    for i in range(batch_size):    
        for j in range(z):
            loss = 0
            output_z = output[i:i+1,:,:,:,j]
            label_z = label[i,:,:,:,j]
            
            softmax_output_z = nn.Softmax2d()(output_z)
            logsoftmax_output_z = torch.log(softmax_output_z)
            
            loss = nn.NLLLoss2d()(logsoftmax_output_z,label_z)
            total_loss += loss
            
    return total_loss

def save_check_points(net, check_points_dir, epoch, optimizer):
    iter_num = config['train_batch_size'] * epoch
    if (epoch+1) % config['save_epoch_num'] == 0: #save last
        torch.save(net.state_dict(),Check_Pints_dir + '%s_model.pth'%(str(epoch).zfill(5)))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : iter_num,
            'epoch'    : epoch,
        }, Check_Pints_dir +'%s_optimizer.pth'%(str(epoch).zfill(5)))

def adjust_lr(optimizer, epoch):
    lr = config['lr'] * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def train(epoch, train_loader, model, optimizer):
    iter_num = 0
    train_loss = 0
    for batch_idx, (data, target, indices) in enumerate(train_loader):
        if config['cuda']:
            data, target = data.cuda(), target.cuda()
        data, train_augment = Variable(data), Variable(target)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, train_augment)
        # loss = F.nll_loss(log_p, target)
        # loss = criterion(output_score, target)
        loss.backward()
        optimizer.step()
        # save the checkpoints
        save_check_points(model, Check_Pints_dir, epoch, optimizer)
        train_loss = train_loss + loss.data[0]
        learning_rate = adjust_lr(optimizer, epoch)
        #learning rate decay



    train_loss = train_loss/len(train_loader.dataset)
    return train_loss, learning_rate

        # some train info name batch_index for train_loss = loss.data[0]


def valid(valid_loader, model):
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target, indices in valid_loader:
        if config['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = loss_function(output, target)
        valid_loss = valid_loss + loss.data[0]

    valid_loss /= len(valid_loader.dataset)

    return valid_loss


def train_augment(image, label, indices):
    """include random crop and random flip transpose"""
    # image = Flip_image(image)
    # label = Flip_image(label)

    image, label = Random_crop(image, label, config['crop_size'])
    image = np.expand_dims(image, axis = 0)
    label = np.expand_dims(label, axis = 0)
    input = torch.from_numpy(image).float().div(255)
    label = torch.from_numpy(label).long()
    return input, label, indices


def valid_augment(image, label, indices):
    """include random crop and random flip transpose"""
    image, label = Random_crop(image, label, config['crop_size'])
    image = np.expand_dims(image, axis=0)
    label = np.expand_dims(label, axis=0)
    input = torch.from_numpy(image.copy()).float().div(255)
    label = torch.from_numpy(label).long()
    return input, label, indices


def main():

    initial_checkpoint = None
    pretrain_file = None
    log.open(out_dir+'/log.train.txt', mode='a')
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    # load image data
    train_dataset = ScienceDataset(
                        split='Train_DataSet',
                        Data_dir=Data_Dir,
                        transform=train_augment, mode='train')

    train_loader = DataLoader(
                        train_dataset,
                        sampler=None,
                        shuffle=True,
                        batch_size=config['train_batch_size'],
                        drop_last=config['drop_last'],
                        num_workers=config['num_workers'],
                        pin_memory=config['pin_memory'])

    valid_dataset = ScienceDataset(
                        split='Valid_DataSet',
                        Data_dir=Data_Dir,
                        transform=valid_augment,
                        mode='train')

    valid_loader = DataLoader(
                        valid_dataset,
                        sampler=None,
                        shuffle=True,
                        batch_size=config['valid_batch_size'],
                        drop_last=False,
                        num_workers=config['num_workers'],
                        pin_memory=config['pin_memory'])

    log.write('** dataset setting **\n')
    log.write(
        '\tWIDTH, HEIGHT = %d, %d, %d\n' % (
            config['crop_size'][0],
            config['crop_size'][1],
            config['crop_size'][2]))
    log.write('\ttrain_dataset.split = %s\n' % (len(train_dataset.ids)))
    log.write('\tvalid_dataset.split = %s\n' % (len(valid_dataset.ids)))
    log.write('\tlen(train_dataset)  = %d\n' % (len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n' % (len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n' % (len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n' % (len(valid_loader)))
    log.write('\tbatch_size  = %d\n' % (config['train_batch_size']))
    log.write('\n')

    start_epoch = 0
    start_iteration = 0
    # initial model
    model = UnetGenerator_3d(in_dim=1, out_dim=2, num_filter=4)
    if config['cuda']:
        model.cuda()
    optimizer = optim.SGD(
                        model.parameters(),
                        lr=config['lr'],
                        momentum=config['momentum'])
    # optimizer = optim.Adam(
    #                     model.parameters(),
    #                     lr=config['lr'],
    #                     betas=(0.9, 0.999),
    #                     eps=1e-8,
    #                     weight_decay=0)
    # start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n' % str(optimizer))
    log.write(' momentum=%f\n' % optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n' % str(config['lr']))
    log.write(' images_per_epoch = %d\n\n' % len(train_dataset))
    log.write(' rate    iter   epoch  num   | valid_loss               | \
                train_loss|  time          \n')
    log.write('---------------------\
                ---------------------\
                ------------------\n')

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        net.load_pretrain(pretrain_file, skip)

    log.write('** net setting **\n')
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('\tpretrain_file = %s\n' % pretrain_file)
    log.write('%s\n\n' % (type(model)))
    log.write('\n')

    for epoch in range(1, config['epochs']+1):
        model.train()
        train_loss,learning_rate = train(epoch, train_loader, model, optimizer)
        # train(epoch)
        valid_loss = valid(valid_loader, model)
        log.write('%d k | train_loss')
        log.write('%d k | %0.3f | %0.3f | %0.3f | %0.3f\n' % (\
                config['train_batch_size'] * epoch , epoch,
                train_loss, valid_loss, learning_rate))
        log.write('\n')

if __name__ == "__main__":
    main()
