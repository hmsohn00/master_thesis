import network
import loss
import dataset
import helper

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

import os
from tqdm import tqdm
import numpy as np
import imageio
import pandas as pd
import argparse
import sys
from str2bool import str2bool

################################################################################
#   PARSE COMMANDLINE OPTIONS
################################################################################

parser = argparse.ArgumentParser(description='Train the CycleGAN on tile and label',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-X_path', '--X_path', help="path to domain X", dest='X_path')
parser.add_argument('-Y_path', '--Y_path', help="path to domain X", dest='Y_path')
parser.add_argument('-save_path', '--save_path', help="path to output file", dest='save_path')
parser.add_argument('-n_epochs', '--n_epochs', type=int, default=100, help='Number of epochs', dest='n_epochs')
parser.add_argument('-batch_size', '--batch_size', type=int, default=6, help='Batch size', dest='batch_size')
parser.add_argument('-lr', '--lr', type=float, default=0.0002, help='Learning rate', dest='lr')
# X_lambda_weight : weight for mapping X->Y
# Y_lambda_weight : weight for mapping Y->X
parser.add_argument('-X_lambda_weight', '--X_lambda_weight', type=float, default=10, help='Lambda weight for X', dest='X_lambda_weight')
parser.add_argument('-Y_lambda_weight', '--Y_lambda_weight', type=float, default=10, help='Lambda weight for Y', dest='Y_lambda_weight')
parser.add_argument('-print_every', '--print_every', type=int, default=1, dest='print_every')
parser.add_argument('-is_scheduler', '--is_scheduler', dest='is_scheduler')
parser.add_argument('-is_data_aug', '--is_data_aug', dest='is_data_aug')
parser.add_argument('-scheduler_step', '--scheduler_step', type=int, default=10, dest='scheduler_step')
parser.add_argument('-num_sample', '--num_sample', type=int, default=5, help='Number of samples within one batch', dest='num_sample')

args = parser.parse_args()

# get save path:
if args.save_path != None:
    save_path = args.save_path
    # If save folder does not exist, create it
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path,'images'))
        os.mkdir(os.path.join(save_path,'saved_models'))
        print('Save path is created')
    else:
        print('Save path already exist')
        
    f = open(os.path.join(save_path,'training_details.txt'), "w+")   # 'w' for writing
    print("# Save path: " + save_path, file=f)
else:
    sys.stderr.write("Please specify save path to output!\n")
    sys.exit(2)

# get data path:
if args.X_path != None:
    print("# X_Path: " + args.X_path, file=f)
    X_path = args.X_path
else:
    sys.stderr.write("Please specify path to domain X!\n")
    sys.exit(2)
if args.Y_path != None:
    print("# Y_Path: " + args.Y_path, file=f)
    Y_path = args.Y_path
else:
    sys.stderr.write("Please specify path to domain X!\n")
    sys.exit(2)

# get batch size:
try:
    print("# Batch size: " + str(args.batch_size) , file=f)
    batch_size = int(args.batch_size)
except:
    sys.stderr.write("Please specify batch size!\n")
    sys.exit(2)

# get number of epochs:
try:
    print("# Number of epochs: " + str(args.n_epochs), file=f)
    n_epochs = int(args.n_epochs)
except:
    sys.stderr.write("Please specify number of epochs!\n")
    sys.exit(2)

# get learning rate:
try:
    print("# Learning rate: " + str(args.lr), file=f)
    lr = float(args.lr)
except:
    sys.stderr.write("Please specify learning rate!\n")
    sys.exit(2)

# get lambda weight:
try:
    print("# Lambda weight for X: " + str(args.X_lambda_weight), file=f)
    X_lambda_weight = float(args.X_lambda_weight)
except:
    sys.stderr.write("Please specify Labmda weight for X!\n")
    sys.exit(2)
try:
    print("# Lambda weight for Y: " + str(args.Y_lambda_weight), file=f)
    Y_lambda_weight = float(args.Y_lambda_weight)
except:
    sys.stderr.write("Please specify Labmda weight for Y!\n")
    sys.exit(2)
    
# get print_every
try:
    print("# Print_every: " + str(args.print_every), file=f)
    print_every = int(args.print_every)
except:
    sys.stderr.write("Please specify print_every!\n")
    sys.exit(2)
    
# get if we use scheduler:
try:
    print("# Use scheduler: " + str(args.is_scheduler) , file=f)
    is_scheduler = str2bool(args.is_scheduler)
except:
    sys.stderr.write("Please specify using scheduler!\n")
    sys.exit(2)
    
# get if we use data augmentation:
try:
    print("# Use data augmentation: " + str(args.is_data_aug) , file=f)
    is_data_aug = str2bool(args.is_data_aug)
except:
    sys.stderr.write("Please specify using data augmentation!\n")
    sys.exit(2)

# get scheduler step
try:
    print("# Scheduler step: " + str(args.scheduler_step), file=f)
    scheduler_step = int(args.scheduler_step)
except:
    sys.stderr.write("Please specify scheduler step!\n")
    sys.exit(2)
    
# get number of epochs:
try:
    print("# Number of samples: " + str(args.num_sample), file=f)
    num_sample = int(args.num_sample)
except:
    sys.stderr.write("Please specify number of samples!\n")
    sys.exit(2)
f.close()

################################################################################
#   Get data loader, create model, optimizer, scheduler
################################################################################

# get beta for Adam optimizer
beta1=0.5
beta2=0.999

# get data set
if is_data_aug:
    # get transformation
    transform_X = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_Y = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    
    X_dset = dataset.HistData(data_path=X_path, transform=transform_X)
    Y_dset = dataset.HistData(data_path=Y_path, transform=transform_Y)
else:
    X_dset = dataset.HistData(data_path=X_path)
    Y_dset = dataset.HistData(data_path=Y_path)

# get data loader for X
X_train_size = int(0.9 * len(X_dset))
X_test_size = len(X_dset) - X_train_size
X_train_dset, X_test_dset = random_split(X_dset, [X_train_size, X_test_size])
X_train_dataloader = DataLoader(X_train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
X_test_dataloader = DataLoader(X_test_dset, batch_size=batch_size, shuffle=True)
torch.save(X_test_dataloader, os.path.join(save_path, 'X_test_dataloader.pth'))

# get data loader for Y
Y_train_size = int(0.9 * len(Y_dset))
Y_test_size = len(Y_dset) - Y_train_size
Y_train_dset, Y_test_dset = random_split(Y_dset, [Y_train_size, Y_test_size])
Y_train_dataloader = DataLoader(Y_train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
Y_test_dataloader = DataLoader(Y_test_dset, batch_size=batch_size, shuffle=True)
torch.save(Y_test_dataloader, os.path.join(save_path, 'Y_test_dataloader.pth'))

# create models
G_XtoY, G_YtoX, D_X, D_Y, device = network.create_model()


# Get generator parameters
g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())

# Create optimizers for the generators and discriminators
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])

# scheduler
if is_scheduler:
    g_lr_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=scheduler_step, gamma=0.1, last_epoch=-1)
    d_x_lr_scheduler = torch.optim.lr_scheduler.StepLR(d_x_optimizer, step_size=scheduler_step, gamma=0.5, last_epoch=-1)
    d_y_lr_scheduler = torch.optim.lr_scheduler.StepLR(d_y_optimizer, step_size=scheduler_step, gamma=0.5, last_epoch=-1)

# keep track of losses over time
losses = []

test_iter_X = iter(X_test_dataloader)
test_iter_Y = iter(Y_test_dataloader)

# Get some fixed data from domains X and Y for sampling. These are images that are held
# constant throughout training, that allow us to inspect the model's performance.
test_images_X = test_iter_X.next()
test_images_Y = test_iter_Y.next()

# batches per epoch
iter_X = iter(X_train_dataloader)
iter_Y = iter(Y_train_dataloader)
batches_per_epoch = min(len(iter_X), len(iter_Y))

################################################################################
#   Training CycleGAN
################################################################################

every_batch = int((min(len(X_train_dataloader), len(Y_train_dataloader)))/num_sample)
df = pd.DataFrame(columns=['epoch','batch','d_x_loss','d_y_loss','g_total_loss'])

for epoch in range(1, n_epochs+1):
    
    epoch_d_x_loss = 0
    epoch_d_y_loss = 0
    epoch_g_total_loss = 0
    
    with tqdm(total=Y_train_size, desc=f'Epoch {epoch}/{n_epochs+1}', unit='img') as pbar:
    
        # Reset iterators for each epoch
        iter_X = iter(X_train_dataloader)
        iter_Y = iter(Y_train_dataloader)

        for idx in range(batches_per_epoch):

            # Reset iterators for each epoch
            images_X = iter_X.next()
            images_Y = iter_Y.next()

            # move images to GPU if available (otherwise stay on CPU)
            images_X = images_X.to(device)
            images_Y = images_Y.to(device)

            # ============================================
            #            TRAIN THE DISCRIMINATORS
            # ============================================

            ##   First: D_X, real and fake loss components   ##

            # Train with real images
            d_x_optimizer.zero_grad()

            # 1. Compute the discriminator losses on real images
            out_x = D_X(images_X)
            D_X_real_loss = loss.real_mse_loss(out_x)

            # Train with fake images

            # 2. Generate fake images that look like domain X based on real images in domain Y
            fake_X = G_YtoX(images_Y)

            # 3. Compute the fake loss for D_X
            out_x = D_X(fake_X)
            D_X_fake_loss = loss.fake_mse_loss(out_x)

            # 4. Compute the total loss and perform backprop
            d_x_loss = D_X_real_loss + D_X_fake_loss
            d_x_loss.backward()
            d_x_optimizer.step()
            epoch_d_x_loss += d_x_loss.item()

            ##   Second: D_Y, real and fake loss components   ##

            # Train with real images
            d_y_optimizer.zero_grad()

            # 1. Compute the discriminator losses on real images
            out_y = D_Y(images_Y)
            D_Y_real_loss = loss.real_mse_loss(out_y)

            # Train with fake images

            # 2. Generate fake images that look like domain Y based on real images in domain X
            fake_Y = G_XtoY(images_X)

            # 3. Compute the fake loss for D_Y
            out_y = D_Y(fake_Y)
            D_Y_fake_loss = loss.fake_mse_loss(out_y)

            # 4. Compute the total loss and perform backprop
            d_y_loss = D_Y_real_loss + D_Y_fake_loss
            d_y_loss.backward()
            d_y_optimizer.step()
            epoch_d_y_loss += d_y_loss.item()

            # =========================================
            #            TRAIN THE GENERATORS
            # =========================================

            ##    First: generate fake X images and reconstructed Y images    ##
            g_optimizer.zero_grad()

            # 1. Generate fake images that look like domain X based on real images in domain Y
            fake_X = G_YtoX(images_Y)

            # 2. Compute the generator loss based on domain X
            out_x = D_X(fake_X)
            g_YtoX_loss = loss.real_mse_loss(out_x)

            # 3. Create a reconstructed y
            # 4. Compute the cycle consistency loss (the reconstruction loss)
            reconstructed_Y = G_XtoY(fake_X)
            # print(images_Y.shape) #[8, 3, 215, 215]
            # print(reconstructed_Y.shape) #[8, 3, 208, 208]
            reconstructed_y_loss = loss.cycle_consistency_loss(images_Y, reconstructed_Y, lambda_weight=Y_lambda_weight)

            ##    Second: generate fake Y images and reconstructed X images    ##

            # 1. Generate fake images that look like domain Y based on real images in domain X
            fake_Y = G_XtoY(images_X)

            # 2. Compute the generator loss based on domain Y
            out_y = D_Y(fake_Y)
            g_XtoY_loss = loss.real_mse_loss(out_y)

            # 3. Create a reconstructed x
            # 4. Compute the cycle consistency loss (the reconstruction loss)
            reconstructed_X = G_YtoX(fake_Y)
            reconstructed_x_loss = loss.cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=X_lambda_weight)

            # 5. Add up all generator and reconstructed losses and perform backprop
            g_total_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_y_loss + reconstructed_x_loss
            g_total_loss.backward()
            g_optimizer.step()
            epoch_g_total_loss += g_total_loss.item()
            
            pbar.set_postfix(**{'loss_G': epoch_g_total_loss, 'loss_DX': epoch_d_x_loss, 'loss_DY': epoch_d_y_loss})
            pbar.update(batch_size)
            
            # save losses
            df.loc[df.shape[0]] = [int(epoch), int(idx), d_x_loss.item(), d_y_loss.item(), g_total_loss.item()]
            df.to_csv(os.path.join(save_path, 'losses.csv'), encoding='utf-8', index=False)

            # generate sample
            if epoch % print_every == 0:
                if idx % every_batch == 0:
                    filename = os.path.join(save_path,'images', 'epoch'+str(epoch)+'_'+str(idx))
                    X_fake = G_XtoY(images_X)
                    grid_image_real = torchvision.utils.make_grid(images_X.cpu())
                    grid_image_fake = torchvision.utils.make_grid(X_fake.cpu())
                    grid_image = torch.cat((grid_image_real, grid_image_fake), 1)
                    path = filename + '_' + 'XtoY.jpg'
                    torchvision.utils.save_image(grid_image, path)
                    print('Saved {}'.format(path))

                    Y_fake = G_YtoX(images_Y)
                    grid_image_real = torchvision.utils.make_grid(images_Y.cpu())
                    grid_image_fake = torchvision.utils.make_grid(Y_fake.cpu())
                    grid_image = torch.cat((grid_image_real, grid_image_fake), 1)
                    path = filename + '_' + 'YtoX.jpg'
                    torchvision.utils.save_image(grid_image, path)
                    print('Saved {}'.format(path))
                    
                    # save models after certein epoch
                    if epoch > 2 :
                        helper.checkpoint(epoch, idx, G_XtoY, G_YtoX, D_X, D_Y, save_path)
                        
        # Print the log info
        if epoch % print_every == 0:
            losses.append((epoch_d_x_loss, epoch_d_y_loss, epoch_g_total_loss))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                epoch, n_epochs, epoch_d_x_loss, epoch_d_y_loss, epoch_g_total_loss))
            
        # update scheduler
        if is_scheduler:
            g_lr_scheduler.step()
            d_x_lr_scheduler.step()
            d_y_lr_scheduler.step()

            