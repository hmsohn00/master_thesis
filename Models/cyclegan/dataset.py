import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
import torch
from PIL import Image
import os
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torchvision.transforms import ToTensor
from torchvision import transforms
import pandas as pd


class HistData(torch.utils.data.Dataset):
    def __init__(self, data_path, im_size=512,
                 transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):

        self.transform = transform
        self.images_path = data_path # + '/'
        self.images_name = os.listdir(self.images_path)
        self.im_size = im_size
        
        # remove not png files
        is_delete = False
        for idx in range(len(self.images_name)):
            _, ext = os.path.splitext(self.images_name[idx])
            if ext != '.png':
                print('idx:',idx,'-',self.images_name[idx], 'is deleted')
                is_delete = True
                delete_idx = idx
        if is_delete:
            self.images_name.pop(delete_idx)

    def __len__(self):
        return len(self.images_name)
    
    def __getitem__(self, idx):

        image_path = os.path.join(self.images_path, self.images_name[idx])
        image = Image.open(image_path)

        if len(image.getbands()) == 4:  # Fix in png.
            # original code
            im2arr = np.array(image)
            image = Image.fromarray(im2arr[:, :, :-1])
            
        image = image.resize((self.im_size, self.im_size),Image.BILINEAR)
        X = self.transform(image)

        return X
