# CycleGANs for Tissue Segmentation
The source code used for master's thesis.

### Data
The data used for the thesis can be downloaded from [GTEx Portal](https://www.gtexportal.org/home/). We downloaded 594 subject's liver and pancreas histology images from GTEx portal. To extract tiles from one `.svs` file, `openslide` package is required. The information about the package can be found here: [Openslide Python](https://openslide.org/api/python/). Due to the size matter, the image tiles are not presented in this repository now. 

### Annotation model
The jupyter notebook for creating simulated label images described in the thesis can be found in `Annotation_model` folder.

- `liver_fat_annotation` contains functions and scripts for generating simulated labels for fat segmentation of liver tissue
- `pancreas_annotation` contains functions and scripts for generating simulated labels for pancreas tissue. It contains functions generating boundary of cell, fat, duct, and islet. One can control the parameter in the notebook to control the size and distribution of each annotation in the pancreas.

### Models
1. CycleGAN 

The scripts for training CycleGAN are in folder `Models/cyclegan/`. To train the CycleGAN, one can run the script called `train_cyclegan.py`, for example:
```
python3 train_cyclegan.py \
  -X_path /path/to/histology/images/ \
  -Y_path /path/to/label/images/ \
  -save_path /path/to/save/path/for/results/ \
```
For `train_cyclegan.py`, one should define `X_path`,`Y_path`, and `save_path`. `X_path` is the path for the histology images, and `Y_path` is for the label images. `save_path` is the path for saving the generated samples or trained model during the training. Other parameters are set with default values, but users can be chosen based on their need.

2. U-net

The scripts for training U-net are in folder `Models/unet`. To train the U-net, one can run the script called `train_unet.py`, for example:
```
python3 train_unet.py \
  -dir_img /path/to/histology/images/ \
  -dir_mask /path/to/mask/images/ \
  -dir_checkpoint /path/to/save/path/for/results/ \
```
For `train_unet.py`, one should define `dir_img`,`dir_mask`, and `dir_checkpoint`. `dir_img` is the path for the histology images, and `dir_mask` is for the aligned mask images. Note that in the path `dir_img`,`dir_mask`, the aligned images should have same file name. `save_path` is the path for saving the generated samples or trained model during the training. Other parameters are set with default values, but users can be chosen based on their need.


##### More updates will follow
