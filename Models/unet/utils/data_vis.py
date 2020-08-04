from PIL import Image
import numpy as np
import glob

import argparse
import logging
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision

from unet import UNet
from utils.dataset import BasicDataset


def print_arg(args):
    for arg in vars(args):
        print(arg,':', getattr(args, arg))
        

def get_output_filenames(args):
    
    num_in_model = re.findall('\d+', args.model)
    epoch = num_in_model[-1]
    
    in_files_path = args.input_folder
    in_files = os.listdir(in_files_path)
    out_files = []

    if not args.output:
        for idx, f in enumerate(in_files):
            pathsplit = os.path.splitext(f)
            if pathsplit[1] == '.png':
                out_files.append("{}_OUT{}{}".format(pathsplit[0],epoch,pathsplit[1]))
            else:
                in_files.pop(idx)
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return in_files, out_files


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
