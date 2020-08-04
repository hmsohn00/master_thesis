# helper functions for saving sample data and models

import os
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
from skimage.color import rgb2gray
import openslide
from openslide import deepzoom
from PIL import Image
from torchvision import transforms
import torch


def preprocess(pil_img):
        
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


def get_cell_mask(tile):
    
    im = np.array(tile)
    im = rgb2gray(im)
    
    cell_mask = im > 0.9
    cell_mask = ndimage.binary_fill_holes(cell_mask)
    cell_mask = ndimage.binary_fill_holes(~cell_mask)
    
    return cell_mask
    

def get_liver_fat_mask(tile, cell_mask, net, device):

    if tile.size != (512,512) or sum(sum(cell_mask)) < 20:
        mask = np.zeros((tile.size[1],tile.size[0]),dtype=bool)
    else:
        img = preprocess(tile)
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        mask = net(img)
        mask = np.transpose(mask[0].detach().cpu().numpy(), (1, 2, 0))

        mask = (mask[:,:,0]>0)^(mask[:,:,2]>0)
        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_erosion(mask,iterations=1)
        mask = ndimage.binary_dilation(mask,iterations=1)
        mask[cell_mask==False] = False
    
    return mask


def get_fat_mask(tile, fat_net, device):

    img = preprocess(tile)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    mask = fat_net(img)
    mask = mask.squeeze(0)
    mask = mask.squeeze(0)

    mask = (mask > 0.01)#.float()
    mask = mask.cpu().data.numpy()
    mask = ndimage.binary_erosion(mask, iterations=8)
    mask = morphology.remove_small_objects(mask, min_size=150, connectivity=2, in_place=True)
    mask = ndimage.binary_dilation(mask, iterations=8)
    mask = ndimage.binary_fill_holes(mask)
    
    return mask


def get_islet_mask(tile, islet_net, device):

    img = preprocess(tile)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    mask = islet_net(img)
    mask = mask.squeeze(0)
    mask = mask.squeeze(0)

    mask = (mask > 0.5)#.float()
    mask = mask.cpu().data.numpy()
    mask = morphology.remove_small_objects(mask, min_size=200, connectivity=1, in_place=True)
    mask = ndimage.binary_fill_holes(mask)
    
    return mask


def get_duct_mask(tile, duct_net, device):

    img = preprocess(tile)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    mask = duct_net(img)
    mask = mask.squeeze(0)
    mask = mask.squeeze(0)

    mask = (mask > 0.3)#.float()
    mask = mask.cpu().data.numpy()    
    mask = morphology.remove_small_objects(mask, min_size=150, connectivity=1, in_place=True)
    mask = ndimage.binary_fill_holes(mask)
    
    return mask


def get_vessel_mask(tile, duct_net, device):

    img = preprocess(tile)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    mask = duct_net(img)
    mask = mask.squeeze(0)
    mask = mask.squeeze(0)

    mask = (mask > 0.5)#.float()
    mask = (mask > mask_threshold)#.float()
    mask = mask.cpu().data.numpy()
    mask = morphology.remove_small_objects(mask, min_size=150, connectivity=1, in_place=True)
    
    return mask


def get_liver_label_matrix(tile, net, device):
    
    cell_mask = get_cell_mask(tile)
    fat_mask = get_liver_fat_mask(tile, cell_mask, net, device)

    new_label = np.empty(shape=(tile.size[1],tile.size[0],2))
    new_label[:,:,0] = cell_mask 
    new_label[:,:,1] = fat_mask 

    # remove background 1 when labeled as other
    for i in range(new_label.shape[0]):
        for j in range(new_label.shape[1]):
            if sum(new_label[i,j,1:]) == 1:
                new_label[i,j,0] = 0
    
    return new_label


def get_label_matrix(tile, fat_net, duct_net, islet_net, vessel_net, device):
    
    cell_mask = get_cell_mask(tile)
    
    if sum(sum(cell_mask)) < 20:
        new_label = np.zeros(shape=(tile.size[1],tile.size[0],5))
    else:
        fat_mask = get_fat_mask(tile, fat_net, device)
        duct_mask = get_fat_mask(tile, duct_net, device)
        islet_mask = get_fat_mask(tile, islet_net, device)
        vessel_mask = get_fat_mask(tile, vessel_net, device)
        
        if sum(sum(cell_mask))/cell_mask.size > 0.003:
            duct_mask[cell_mask == False] = False
            islet_mask[cell_mask == False] = False
            vessel_mask[cell_mask == False] = False
        else:
            duct_mask[cell_mask == False] = False
            fat_mask[cell_mask == False] = False
            islet_mask[cell_mask == False] = False
            vessel_mask[cell_mask == False] = False

        # if duct and fat intersect, remove fat
        fat_mask[duct_mask == fat_mask] = False

        # if duct and islet intersect, remove islet
        islet_mask[duct_mask == islet_mask] = False

        # if duct and vessel intersect, remove duct
        duct_mask[duct_mask == vessel_mask] = False

        # if vessel and islet intersect, remove vessel
        vessel_mask[islet_mask == vessel_mask] = False

        # if vessel and fat intersect, remove vessel
        vessel_mask[fat_mask == vessel_mask] = False

        # if islet and fat intersect, remove fat
        fat_mask[fat_mask==islet_mask] = False

    
        new_label = np.empty(shape=(tile.size[1],tile.size[0],5))
        new_label[:,:,0] = cell_mask 
        new_label[:,:,1] = fat_mask 
        new_label[:,:,2] = vessel_mask 
        new_label[:,:,3] = islet_mask
        new_label[:,:,4] = duct_mask

        # remove background 1 when labeled as other
        for i in range(new_label.shape[0]):
            for j in range(new_label.shape[1]):
                if sum(new_label[i,j,1:]) == 1:
                    new_label[i,j,0] = 0
    
    return new_label


def pil_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def pil_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
    
    
def np_concat_h(a,b):
    if a.shape[0] != b.shape[0]:
        if a.shape[0] > b.shape[0]:
            a = np.delete(a, 1, 0)
        else:
            b = np.delete(b, 1, 0)
    tmp = np.concatenate((a, b), axis=1)
    return tmp


def np_concat_v(a,b):

    if a.shape[1] != b.shape[1]:
        if a.shape[1] > b.shape[1]:
            a = np.delete(a, 1, 1)
        else:
            b = np.delete(b, 1, 1)
    tmp = np.concatenate((a, b), axis=0)
    return tmp


def get_liver_area(total_label):
    
    cell_mask = total_label[:,:,0]
    fat_mask = total_label[:,:,1]
   
    acinar_area = sum(sum(cell_mask))
    fat_area = sum(sum(fat_mask))

    total_area = acinar_area + fat_area 
    
    return total_area, acinar_area, fat_area


def get_area(total_label):
    
    cell_mask = new_label[:,:,0]
    fat_mask = new_label[:,:,1]
    vessel_mask = new_label[:,:,2]
    islet_mask = new_label[:,:,3]
    duct_mask = new_label[:,:,4]
    
    acinar_area = sum(sum(cell_mask))
    fat_area = sum(sum(fat_mask))
    vessel_area = sum(sum(vessel_mask))
    islet_area = sum(sum(islet_mask))
    duct_area = sum(sum(duct_mask))
    total_area = acinar_area + fat_area + vessel_area + islet_area + duct_area
    
    return total_area, acinar_area, fat_area, vessel_area, islet_area, duct_area


def overlay_liver_mask_tile(tile, label):
    
    t = 0.7

    tile_im = np.array(tile)
    gray_tile = rgb2gray(tile_im)

    # label = label*255
    cell_mask = label[:,:,0]
    fat_mask = label[:,:,1]

    im = np.zeros(shape=(tile.size[1],tile.size[0],3))

    im[:,:,0] = t*gray_tile + (1-t)*(fat_mask)
    im[:,:,1] = t*gray_tile + (1-t)*(cell_mask)
    im[:,:,2] = t*gray_tile + (1-t)*(cell_mask)
    
    return im


def overlay_mask_tile(tile, label):
    
    t = 0.5

    tile_im = np.array(tile)
    gray_tile = rgb2gray(tile_im)

    # label = label*255
    cell_mask = label[:,:,0]
    fat_mask = label[:,:,1]
    vessel_mask = label[:,:,2]
    islet_mask = label[:,:,3]
    duct_mask = label[:,:,4]

    im = np.zeros(shape=(tile.size[1],tile.size[0],3))

    im[:,:,0] = t*gray_tile + (1-t)*(fat_mask+duct_mask)
    im[:,:,1] = t*gray_tile + (1-t)*(vessel_mask+duct_mask+cell_mask)
    im[:,:,2] = t*gray_tile + (1-t)*(islet_mask+cell_mask)
    
    return im
    
    
def overlay_mask_total_liver(data_path, label_path, im_name):
    
    label = np.load(os.path.join(label_path,im_name+'.npy'))

    wsi = openslide.open_slide(os.path.join(data_path,im_name+'.svs'))
    dzi = openslide.deepzoom.DeepZoomGenerator(wsi, tile_size=512, overlap = 0, limit_bounds = False)

    # extract tiles
    x_tile = dzi.level_tiles[14][0] # column
    y_tile = dzi.level_tiles[14][1] # row
    num_tiles = x_tile*y_tile
    
    # go through each column
    for i in range(x_tile):
        # go thtough each row
        for j in range(y_tile):
            tile = dzi.get_tile(14,(i,j))                
            if j == 0:
                col_im = tile
            else:
                col_im = pil_concat_v(col_im, tile)
        if i == 0:
            total_im = col_im
        else:
            total_im = pil_concat_h(total_im, col_im)
    
    if label.shape[0] != total_im.size[1]:
        if label.shape[0] > total_im.size[1]:
            label = np.delete(label, 1, 0)

    overlay_im = overlay_liver_mask_tile(total_im, label)

    return overlay_im


def overlay_mask_total(data_path, label_path, im_name):
    
    label = np.load(os.path.join(label_path,im_name+'.npy'))

    wsi = openslide.open_slide(os.path.join(data_path,im_name+'.svs'))
    dzi = openslide.deepzoom.DeepZoomGenerator(wsi, tile_size=512, overlap = 0, limit_bounds = False)

    # extract tiles
    x_tile = dzi.level_tiles[14][0] # column
    y_tile = dzi.level_tiles[14][1] # row
    num_tiles = x_tile*y_tile
    
    # go through each column
    for i in range(x_tile):
        # go thtough each row
        for j in range(y_tile):
            tile = dzi.get_tile(14,(i,j))                
            if j == 0:
                col_im = tile
            else:
                col_im = pil_concat_v(col_im, tile)
        if i == 0:
            total_im = col_im
        else:
            total_im = pil_concat_h(total_im, col_im)
    
    if label.shape[0] != total_im.size[1]:
        if label.shape[0] > total_im.size[1]:
            label = np.delete(label, 1, 0)

    overlay_im = overlay_mask_tile(total_im, label)

    return overlay_im