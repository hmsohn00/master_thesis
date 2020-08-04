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
  -save_path /path/to/save/path/for/results/ 
```
For `train_cyclegan.py`, one should define `X_path`,`Y_path`, and `save_path`. `X_path` is the path for the histology images, and `Y_path` is for the label images. `save_path` is the path for saving the generated samples or trained model during the training. Other parameters are set with default values, but users can be chosen based on their need.

2. U-net

The scripts for training U-net are in folder `Models/unet`. To train the U-net, one can run the script called `train_unet.py`, for example:
```
python3 train_unet.py \
  -dir_img /path/to/histology/images/ \
  -dir_mask /path/to/mask/images/ \
  -dir_checkpoint /path/to/save/path/for/results/ 
```
For `train_unet.py`, one should define `dir_img`,`dir_mask`, and `dir_checkpoint`. `dir_img` is the path for the histology images, and `dir_mask` is for the aligned mask images. Note that in the path `dir_img`,`dir_mask`, the aligned images should have same file name. `save_path` is the path for saving the generated samples or trained model during the training. Other parameters are set with default values, but users can be chosen based on their need.

### Prediction
In folder `Prediction`, the scripts for predicting the whole slide image of pancreas and liver. The pretrained models for fat in liver and different tissue compartments in pancreas is saved in folder `Prediction/models`.

To predict fat in the liver, run script `liver_predict.py`, for example:
```
python3 liver_predict.py \
  -data_path /path/to/whole/slide/histology/images/ 
``` 

To predict different tissue compartments(acinar, fat, vessel, islet, duct) in the pancreas, run script `pancreas_predict.py`, for example:
```
python3 pancreas_predict.py \
  -data_path /path/to/whole/slide/histology/images/ 
```

The `data_path` is the path that contains all 594 histology images of liver or pancreas. Due to size matter, I devided the images into 10 batches, so the script can be run parallel simultaneously by choosing different batch number. These prediction script will save the numpy boolean array as segmentation mask for each whole slide image.

The saved mask can be overlaid to the original histolgy images by using script `get_overlay_im.py` by
```
python3 get_overlay_im.py \
  -data_path /path/to/whole/slide/histology/images/ \
  -label_path /path/to/saved/segmentation/mask/ \
  -im_name /name/of/the/image/wanted/to/make/overlaid/image/ \
  -is_liver True
```
If the image is liver, then parameter `is_liver` is True, otherwise if it is pancreas, then False.

##### More updates will follow
