def check_train_augment(image, label, indices):
    """include random crop and random flip transpose"""
    # image = Flip_image(image)
    # label = Flip_image(label)
    image, label = Random_crop(image, label, config['crop_size'])
    return image, label, indices


def check_valid_augment(image, label, indices):
    """include random crop and random flip transpose"""
    image, label = Random_crop(image, label, config['crop_size'])
    return image, label, indices


def check_images_labels(images, labels):
    # catch images and labels
    # batch, channel, width, height, depth = images.cpu().numpy().shape
    plt.imshow(data[0, 0, :, :].cpu().numpy())
    plt.title('data')
    plt.pause(0.3)
    plt.imshow(target[0, :, :].cpu().numpy())
    plt.title('target')
    plt.pause(0.3)


def check_dataloader():
    """chech dataloader for dataset"""
    train_dataset = ScienceDataset(
                            split='Train_DataSet',
                            Data_dir=Data_Dir
                            ,
                            transform=check_train_augment,
                            mode='train')

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
                                transform=check_valid_augment,
                                mode='train')

    valid_loader = DataLoader(
                        valid_dataset,
                        sampler=None,
                        shuffle=True,
                        batch_size=config['valid_batch_size'],
                        drop_last=config['drop_last'],
                        num_workers=config['num_workers'],
                        pin_memory=config['pin_memory'])

    # check train_loader:
    for batch_image, batch_label, indices in train_loader:
        plt.imshow(batch_image[0,:,:,48])
        plt.pause(0.1)
        plt.imshow(batch_label[0,:,:,48])
        plt.pause(0.1)
        print(batch_image.shape, batch_label.shape)
    print(indices)
    # check valid loader
    # for batch_image, batch_label, indices in valid_loader:
    #     print(batch_image.shape, batch_label.shape)

    print('sucuess!')
