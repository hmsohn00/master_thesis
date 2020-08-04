import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import morphology
from skimage.io import imsave
from scipy import ndimage
import os
import glob
import openslide
from openslide import deepzoom
import importlib
import torch
from PIL import Image
import re
from torchvision import transforms
import torchvision
import GPUtil
import argparse
import timeit 

from networks.cyclegan import create_model 
from mask import *


#=============================================================
#   PARSE COMMANDLINE OPTIONS
#=============================================================
parser = argparse.ArgumentParser()
parser.add_argument('-data_path', '--data_path',  help="path to data file")
parser.add_argument('-save_path', '--save_path',  help="path to output file", default='./results/')
parser.add_argument('-model_path', '--model_path', help="Path contains all the models", default='./models/')
parser.add_argument('-batch_num', '--batch_num', help="Batch number, 1 to 10", default = 1)
args = parser.parse_args()

# get data path:
if args.data_path != None:
    print("# Data path: " + args.data_path )
    data_path = args.data_path
else:
    sys.stderr.write("Please specify path to data!\n")
    sys.exit(2)

# get save path:
if args.save_path != None:
    print("# Save path: " + args.save_path )
    save_path = args.save_path
    mask_path = os.path.join(save_path,'mask')
    masked_image_path = os.path.join(save_path,'masked_image')
    # make save_path directory
    try:                    
        os.mkdir(save_path)
        os.mkdir(mask_path)
        os.mkdir(masked_image_path)
        print('Created save directory')
    except OSError:
        print('Save directory already exists')
        pass
else:
    sys.stderr.write("Please specify save path to output!\n")
    sys.exit(2)

# get model path:
try:
    print("# Model path: " + str(args.model_path))
    model_path = args.model_path
except:
    sys.stderr.write("Problem defining pretrained model path!\n")
    sys.exit(2)
    
# get batch number
if args.batch_num != None:
    print("# Batch number: " + str(args.batch_num) )
    batch_num = int(args.batch_num)
else:
    sys.stderr.write("Please specify batch num!\n")
    sys.exit(2)
    
    
#=============================================================
#   LOAD MODELS AND DEVICE
#=============================================================

# load pre-trained model
net, _, _, _, device = create_model()
net.to(device=device)
net.load_state_dict(torch.load(os.path.join(model_path,'liver.pkl'), map_location=device))
print("Model loaded !")


#=============================================================
#   PREDICTION 
#=============================================================
# total_time = 0
df_name = os.path.join(save_path, str(batch_num) + '_liver_prediction.csv')

if os.path.isfile(df_name):
    df = pd.read_csv(df_name)
else:
    df = pd.DataFrame(columns=['Sample_ID','Total_Area','Cell_Area','Fat_Area'])

file_list = glob.glob(data_path + '*.svs')
if batch_num < 10:
    file_list = file_list[(batch_num-1)*60:batch_num*60] 
else:
    file_list = file_list[(batch_num-1)*60:] 

im_size = 512

for idx, file in enumerate(file_list):
    sample_id,_ = os.path.splitext(os.path.basename(file))
    
    if os.path.isfile(os.path.join(mask_path, sample_id+'.npy')):
        print(idx,'/',len(file_list),'-',sample_id,'- already exist')
    else:
        try:
            wsi = openslide.open_slide(file)
            dzi = openslide.deepzoom.DeepZoomGenerator(wsi, tile_size=im_size, overlap = 0, limit_bounds = False)
            print(idx,'/',len(file_list),'-',sample_id)
        except:
            print(idx,'/',len(file_list),'-',sample_id,'- not available')
            continue

        # extract tiles
        x_tile = dzi.level_tiles[14][0] # column
        y_tile = dzi.level_tiles[14][1] # row
        num_tiles = x_tile*y_tile

        if num_tiles > 1000:
            print('Wrong scale for', sample_id)
            continue
        else:
            # go through each column
            for i in range(x_tile):
                # go thtough each row
                for j in range(y_tile):
                    tile = dzi.get_tile(14,(i,j))                
                    label = get_liver_label_matrix(tile, net, device)
                    if j == 0:
                        col_label = label
                    else:
                        col_label = np_concat_v(col_label, label)
                if i == 0:
                    total_label = col_label
                else:
                    total_label = np_concat_h(total_label, col_label)

        # get area of each 
        total_area, cell_area, fat_area = get_liver_area(total_label)

        # save
        df.loc[df.shape[0]] = [sample_id, total_area, cell_area, fat_area]
        df.to_csv(df_name, encoding='utf-8', index=False)

        # save
        np.save(os.path.join(mask_path, sample_id+'.npy'), total_label.astype(bool))

print('Done')