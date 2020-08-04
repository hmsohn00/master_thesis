import numpy as np
import openslide
from openslide import deepzoom
from PIL import Image
from str2bool import str2bool
from mask import *

#=============================================================
#   PARSE COMMANDLINE OPTIONS
#=============================================================
parser = argparse.ArgumentParser()
parser.add_argument('-data_path', '--data_path',  help="path to data file")
parser.add_argument('-label_path', '--label_path',  help="path to label file (.npy)")
parser.add_argument('-save_path', '--save_path',  help="path to output file", default='./results/')
parser.add_argument('-im_name', '--im_name',  help="name of the image want to make overlay plot")
parser.add_argument('-is_liver', '--is_liver', dest='is_liver', help='True if the data is liver, False if the data is pancreas')
args = parser.parse_args()

# get data path:
if args.data_path != None:
    print("# Data path: " + args.data_path)
    data_path = args.data_path
else:
    sys.stderr.write("Please specify path to data!\n")
    sys.exit(2)
    
# get label path:
if args.label_path != None:
    print("# Label path: " + args.label_path)
    label_path = args.label_path
else:
    sys.stderr.write("Please specify path to label!\n")
    sys.exit(2)

# get save path:
if args.save_path != None:
    print("# Save path: " + args.save_path )
    save_path = args.save_path
    # make save_path directory
    try:                    
        os.mkdir(save_path)
        print('Created save directory')
    except OSError:
        print('Save directory already exists')
        pass
else:
    sys.stderr.write("Please specify save path to output!\n")
    sys.exit(2)

# get image name:
try:
    print("# Image name: " + str(args.im_name))
    im_name = args.im_name
except:
    sys.stderr.write("Please specify name of the image!\n")
    sys.exit(2)
    
# chose if the data is liver or pancreas:
try:
    if str2bool(args.is_liver):
        datatype = 'Liver'
    else:
        datatype = 'Pancreas'
    print("# Used data: " + datatype , file=f)
    is_liver = str2bool(args.is_liver)
except:
    sys.stderr.write("Please specify liver or pancreas!\n")
    sys.exit(2)
    
    
#=============================================================
#       Get overlay image
#=============================================================

if is_liver:
    overlay_im = overlay_mask_total_liver(data_path, label_path, im_name)
else:
    overlay_im = overlay_mask_total(data_path, label_path, im_name)
    
# save the image
pil_overlay = Image.fromarray((overlay_im * 255).astype(np.uint8))
pil_overlay.save(os.path.join(save_path, im_name+'.pdf'))