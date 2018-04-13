# -*- coding: utf-8 -*-
# @Author: antigen
# @Date:   2018-04-12 22:39:01
# @Last Modified by:   antigen
# @Last Modified time: 2018-04-13 16:45:00
from DataReader.reader import ScienceDataset
from DataReader.transform import Flip_image, Random_crop
from torch.utils.data import DataLoader
from DataReader.sampler import *
# from Model.model import *
from 
from Model.UnetGenerator_3d import *
from config import config
from common import *




def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        if config['cuda']:
            data, target = data.cuda(), target.cuda()
        if 0:
            check_images_labels(data, target)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        print("data shape: ", data.size())
        print("target shape: ", target.size())
        output_score = model(data)

        # plt.imshow(output[0,0,:,:].data.cpu().numpy())
        # plt.title('output (score)')
        # plt.pause(0.3)

        n, c, h, w, d = output_score.size()

        output = output_score.contiguous().view(-1, c, h, w*d) # h: 60 w*d: 1200
        m = torch.nn.LogSoftmax()
        log_p = m(output)
        print("         maximum val of log_p:", log_p.data.max())
        print("         minimum val of log_p:", log_p.data.min())
        print("output shape 1: ", output.size())

        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p.view(-1, c)
        mask = target >= 0
        target = target[mask]
        print("log_p shape 2: ", log_p.size())
        print("target shape: ", target.size())

        loss = F.nll_loss(log_p, target)
        print("epoch    ----------------------: ", epoch)
        print("batch_idx----------------------: ", batch_idx)
        print("loss     ----------------------: ", loss.data[0])

        if epoch % args.log_interval == 0 and batch_idx == 0:
            #plt.imshow(output_score[0,0,:,:,30].data.cpu().numpy())
            #plt.title('output (score)')
            #plt.pause(0.3)
            imgname = 'epoch' + str(epoch) + 'loss_' + str(loss.data[0]) + '.png'
            plt.imsave(imgname,output_score[0,0,:,:,30].data.cpu().numpy())


        loss.backward()
        optimizer.step()

        if batch_idx % 26 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t score_max: {:.6f}\t score_min: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], output_score[0,0,:,:,30].data.max(), output_score[0,0,:,:,30].data.min()))

def test():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in validation_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        print("data dim", data.size())
        print("target dim", target.size())
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(validation_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))

def train_augment(image, label, indices):
    """include random crop and random flip transpose"""
    image = Flip_image(image)
    label = Flip_image(label)

    image, label = Random_crop(image, label, config['crop_size'])
    image = np.expand_dims(image, axis=0)
    label = np.expand_dims(label, axis=0)
    input = torch.from_numpy(image.copy()).float().div(255)
    return input, label, indices

def valid_augment(image, label, indices):
    """include random crop and random flip transpose"""
    image, label = Random_crop(image, label, config['crop_size'])
    image = np.expand_dims(image, axis=0)
    label = np.expand_dims(label, axis=0)
    input = torch.from_numpy(image.copy()).float().div(255)
    return input, label, indices

def main(Dataset_dir, Split_dir):
    #load image data
    split_labels = os.path.join(Split_dir, 'labels')
    split_images = os.path.join(Split_dir, 'images')

    train_dataset = ScienceDataset(split_images = split_images, 
        split_labels = split_labels, Data_dir = Dataset_dir,
        transform=train_augment, mode='train')

    train_loader  = DataLoader(
                        train_dataset,
                        sampler = None,
                        shuffle = True,
                        batch_size  = config['train_batch_size'],
                        drop_last   = config['drop_last'],
                        num_workers = config['num_workers'],
                        pin_memory  = config['pin_memory'])

    valid_dataset = ScienceDataset(split_images = split_images, 
        split_labels = split_labels, Data_dir = Dataset_dir,
        transform=valid_augment, mode='train')

    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler = None,
                        shuffle = True,
                        batch_size  = config['valid_batch_size'],
                        drop_last   = config['drop_last'],
                        num_workers = config['num_workers'],
                        pin_memory  = config['pin_memory'])
    start_epoch = 0
    start_iteration = 0
    vgg16 = VGG16(pretrained=True)
    model = Net()
    # model.copy_params_from_vgg16(vgg16)
    if args.cuda:
        model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr= lr, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0)

    for epoch in range(1, epochs+1):
        print("epoch: ",epoch)
        model.train()
        train(epoch)
        # test()

def check_network():
    vgg16 = VGG16(pretrained=True)
    model = Net()
    print(VGG16)
    return True

def check_images_labels(images, labels):
    #catch images and labels
    # batch, channel, width, height, depth = images.cpu().numpy().shape
    plt.imshow(data[0,0,:,:].cpu().numpy())
    plt.title('data')
    plt.pause(0.3)
    plt.imshow(target[0,:,:].cpu().numpy())
    plt.title('target')
    plt.pause(0.3)

def check_dataloader(Dataset_dir, Split_dir):
    """chech dataloader for dataset"""
    split_labels = os.path.join(Split_dir, 'labels')
    split_images = os.path.join(Split_dir, 'images')

    train_dataset = ScienceDataset(split_images = split_images, 
        split_labels = split_labels, Data_dir = Dataset_dir,
        transform=train_augment, mode='train')

    train_loader  = DataLoader(
                        train_dataset,
                        sampler = None,
                        shuffle = True,
                        batch_size  = config['train_batch_size'],
                        drop_last   = config['drop_last'],
                        num_workers = config['num_workers'],
                        pin_memory  = config['pin_memory'])

    valid_dataset = ScienceDataset(split_images = split_images, 
        split_labels = split_labels, Data_dir = Dataset_dir,
        transform=valid_augment, mode='train')

    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler = None,
                        shuffle = True,
                        batch_size  = config['valid_batch_size'],
                        drop_last   = config['drop_last'],
                        num_workers = config['num_workers'],
                        pin_memory  = config['pin_memory'])

    # check train_loader:
    # for batch_image, batch_label, indices in train_loader:
    #     print(batch_image.shape, batch_label.shape)
    #     # print(indices)
    #check valid loader
    # for batch_image, batch_label, indices in valid_loader:
    #     print(batch_image.shape, batch_label.shape)

    print('sucuess!')

if __name__ == "__main__":
    Split_dir = '/home/liuh/Documents/3D_pytorch/build/DataReader/split'
    Dataset_dir = '/home/liuh/Desktop/datasets/data'
    # check_dataloader(Dataset_dir, Split_dir)
    check_network()
	# main(Dataset_dir, Split_dir)
