import argparse
import logging
import os
import sys
import GPUtil
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import pandas as pd
import random
import torch
import torch.nn as nn
from torch import optim
import torchvision

from unet import UNet
import utils.data_vis
import dice_loss

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split


#=====================================================
#   PARSE COMMANDLINE OPTIONS
#=====================================================

parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-dir_img', '--dir_img',
                    help="Specify the path in which the images are stored")
parser.add_argument('-dir_mask', '--dir_mask', 
                    help="Specify the path in which the masks are stored")
parser.add_argument('-dir_checkpoint', '--dir_checkpoint', 
                    help="Specify the path of checkpoint")
parser.add_argument('-epochs', '--epochs', type=int, default=4000,
                    help='Number of epochs', dest='epochs')
parser.add_argument('-batchsize', '--batchsize', type=int, nargs='?', default=8,
                    help='Batch size', dest='batchsize')
parser.add_argument('-lr', '--lr', type=float, nargs='?', default=0.01,
                    help='Learning rate', dest='lr')
parser.add_argument('-im_size', '--im_size', dest='im_size', type=int, default=512,
                    help='Resize size of the images')
parser.add_argument('-test', '--test', dest='test', type=float, default=10.0,
                    help='Percent of the data that is used as test set (0-100)')
parser.add_argument('-save_cp', '--save_cp', default = True, help='Specify if you want to save models')
parser.add_argument('-device_num', '--device_num', type=int, nargs='?', default=1,
                    help='Number of gpus to use', dest='device_num')
args = parser.parse_args()

utils.data_vis.print_arg(args)


#=====================================================
#   Create checkpoint directory
#=====================================================
if args.save_cp:
    try:
        os.mkdir(args.dir_checkpoint)
        os.mkdir(os.path.join(args.dir_checkpoint,'images'))
        os.mkdir(os.path.join(args.dir_checkpoint,'images','test'))
        print('Created checkpoint directory and image folder')
    except OSError:
        pass
    
    
#=====================================================
#   Create model 
#=====================================================

net = UNet(n_channels=3, n_classes=1, bilinear=True)

base_device = GPUtil.getAvailable(order = "memory")[0] if torch.cuda.is_available() else 'cpu'
device = torch.device(base_device)
net.to(device=device)
print('Model moved to', device)

if args.device_num > 1:
    if torch.cuda.device_count() > 1:
        device_ids = list(range(base_device, base_device+device_num))
        print(device_num, "GPUs will be use:",device_ids)
        net = nn.DataParallel(net, device_ids = device_ids)
    else:
        print('Try to use',args.device_num,'gpus, but we can only use',torch.cuda.device_count(), 'gpu.')

#=====================================================
#   Get data loader 
#=====================================================

dataset = BasicDataset(args.dir_img, args.dir_mask, args.im_size)
test_percent = args.test / 100
n_test = int(len(dataset) * test_percent)
n_train = len(dataset) - n_test
train, test = random_split(dataset, [n_train, n_test])
train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True)
test_loader = DataLoader(test, batch_size=args.batchsize, shuffle=False, drop_last=True)


#=====================================================
#   Get optimizer and learning rate scheduler 
#=====================================================

optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)


#=====================================================
#   Train the network
#=====================================================

print_every = 10
num_sample = 3
every_batch = int(len(train_loader)/num_sample)

df = pd.DataFrame(columns=['epoch','loss','testscore', 'testloss'])
losses = []
testscore = []
testloss = []

for epoch in range(args.epochs):
    epoch_loss = 0
    epoch_testscore = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
        
        # =======Start training all batches=======
        for idx, batch in enumerate(train_loader):
            
            net.train()
            
            imgs = batch['image']
            true_masks = batch['mask']

            assert imgs.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float34 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            
            masks_pred = net(imgs)

            loss = dice_loss.unet_total_loss(masks_pred, true_masks)
            pbar.set_postfix(**{'loss(batch)': loss.item()})
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            # generate sample image
            if epoch % print_every == 0:
                # save the sample image
                if idx % every_batch == 0:
                    im_name = 'epoch' + str(epoch) + '_' + str(idx) + '.png'
                    filename = os.path.join(args.dir_checkpoint,'images', im_name)
                    grid_image_real_img = torchvision.utils.make_grid(imgs)
                    grid_image_real_mask = torchvision.utils.make_grid(true_masks)
                    grid_image_pred_mask = torchvision.utils.make_grid(masks_pred.to(dtype=mask_type))
                    grid_image1 = torch.cat((grid_image_real_img, grid_image_real_mask), 1)
                    grid_image = torch.cat((grid_image1, grid_image_pred_mask), 1)
                    torchvision.utils.save_image(grid_image, filename)
                    print('Saved sample {}'.format(filename))

            pbar.update(imgs.shape[0])
            
        # =======Done training all batches within one epoch=======


        # test round
        for idx, batch in enumerate(test_loader):
            net.eval()
            test_imgs = batch['image'].to(device=device)
            test_true_masks = batch['mask'].to(device=device)
            with torch.no_grad():
                test_masks_pred = net(test_imgs)
            
            epoch_testscore = dice_loss.dice_coef(test_masks_pred, test_true_masks).mean()
            epoch_testloss = dice_loss.unet_total_loss(test_masks_pred, test_true_masks).item()

            # save test sample
            if epoch % print_every == 0:  
                im_name = 'epoch' + str(epoch) + '_test_' + str(idx) + '.png'
                filename = os.path.join(args.dir_checkpoint,'images','test', im_name)

                grid_image_real_img = torchvision.utils.make_grid(test_imgs)
                grid_image_real_mask = torchvision.utils.make_grid(test_true_masks)
                grid_image_pred_mask = torchvision.utils.make_grid(test_masks_pred.to(dtype=mask_type))
                grid_image1 = torch.cat((grid_image_real_img, grid_image_real_mask), 1)
                grid_image = torch.cat((grid_image1, grid_image_pred_mask), 1)
                torchvision.utils.save_image(grid_image, filename)
                print('Saved sample {}'.format(filename))
            

        # append total loss
        losses.append(epoch_loss)

        # append test loss
        testscore.append(epoch_testscore.item())
        testloss.append(epoch_testloss)
        
        # save losses ['epoch','loss','testscore', 'testloss']
        df.loc[df.shape[0]] = [epoch, epoch_loss, epoch_testscore.item(), epoch_testloss]
        df.to_csv(os.path.join(args.dir_checkpoint, 'losses.csv'), encoding='utf-8', index=False)


        # Print the log info
        if epoch % print_every == 0:
            print('Epoch [{:5d}/{:5d}] | epoch_loss: {:6.4f} | epoch_testsocre: {:6.4f}, epoch_testsocre: {:6.4f}'.format(
                epoch, args.epochs, epoch_loss/(idx+1), epoch_testscore, epoch_testloss))
            if args.save_cp:
                torch.save(net.state_dict(),
                        args.dir_checkpoint + f'CP_epoch{epoch}.pth')
                print(f'Checkpoint {epoch + 1} saved !')

# last epoch
print('Epoch [{:5d}/{:5d}] | epoch_loss: {:6.4f} | epoch_testsocre: {:6.4f}, epoch_testsocre: {:6.4f}'.format(
                epoch, args.epochs, epoch_loss, epoch_testscore, epoch_testloss))
if args.save_cp:
    torch.save(net.state_dict(),
            args.dir_checkpoint + f'CP_epoch{epoch}.pth')
    print(f'Checkpoint {epoch + 1} saved !')

