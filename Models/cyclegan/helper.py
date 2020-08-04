# helper functions for saving sample data and models

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# import torch
import torch

# numpy & scipy imports
import numpy as np
import scipy
import scipy.misc
from torchvision import transforms
from PIL import Image


def checkpoint(epoch, idx, G_XtoY, G_YtoX, D_X, D_Y, checkpoint_dir='checkpoints_cyclegan'):
    """
    Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
 
    save_path = os.path.join(checkpoint_dir, 'saved_models')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    filename = os.path.join(save_path, 'epoch'+str(epoch)+'_'+str(idx))
    G_XtoY_path = filename + '_G_XtoY.pkl'
    G_YtoX_path = filename + '_G_YtoX.pkl'
    D_X_path = filename + '_D_X.pkl'
    D_Y_path =filename +  '_D_Y.pkl'
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)
    
    print('Saved models to {}'.format(save_path))


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    x = ((x + 1) * 255 / (2)).astype(np.uint8)  # rescale to 0-255
    return x

    
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def preprocess(pil_img, im_size):

    # resize the image if it is different from im_size
    if not pil_img.size == (im_size,im_size):
        pil_img = pil_img.resize((im_size,im_size))
        
    if len(pil_img.getbands()) == 4:  # Fix in png.
        im2arr = np.array(pil_img)
        pil_img = Image.fromarray(im2arr[:, :, :-1])

    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # convert to tensor and return
    return transform(pil_img)


def unnormalize_images(images):
    """
    Return tensor of images to [0:255] space from [-1:1] for plotting. 
    """
    unnormalized = torch.zeros_like(images)
    t = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5])
    for i, image in enumerate(images):
        unnormalized[i] = t(image)*255
    return unnormalized
