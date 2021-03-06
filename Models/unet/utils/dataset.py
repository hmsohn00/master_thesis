from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms.functional as TF


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, im_size = 512, aug_prob = 0.5):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.im_size = im_size
        self.aug_prob = aug_prob

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, im_size):
        
        # resize the image if it is different from im_size
        if not pil_img.size == (im_size,im_size):
            pil_img = pil_img.resize((im_size,im_size))
        
        # convert PIL image to numpy array
        img_nd = np.array(pil_img)
        
        # expand if it is mask image
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # normalize to [0,1]
        if img_nd.max() > 1:
            img_nd = img_nd / 255
        
        # convert to tensor and return
        return transforms.ToTensor()(img_nd)
    
    @classmethod
    def transform(cls, image, mask, aug_prob):
        
        # Random horizontal flipping
        if random.random() > aug_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > aug_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # random brightness
        if random.random() > aug_prob:
            scale_factor = random.uniform(0.7, 1.3)
            image = TF.adjust_brightness(image, scale_factor)
            
        return image, mask

    def __getitem__(self, i):
        idx = self.ids[i]
        
#         # code only for islet
#         mask_file = os.path.join(self.masks_dir,idx+'.png')
#         img_file = os.path.join(self.imgs_dir,idx+'.png')
#         mask = Image.open(mask_file).convert('L')
#         img = Image.open(img_file)
        
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
             f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        mask = Image.open(mask_file[0]).convert('L')
        img = Image.open(img_file[0])

        # resize to require image size
        mask = mask.resize((self.im_size, self.im_size))
        img = img.resize((self.im_size, self.im_size))
        
        # transform both image and mask for data augmentation
        img, mask = self.transform(img, mask, self.aug_prob)

        # normalize and make datas to tensor
        img = self.preprocess(img, self.im_size)
        mask = self.preprocess(mask, self.im_size)

        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        return {'image': img.type(dtype), 'mask': mask.type(dtype)}
