import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F


def dice_coef(pred, target, smooth = 1.):  
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    return (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
     
    
def dice_coef_loss(pred, target, smooth = 1.):
    loss = 1 - dice_coef(pred, target, smooth)
    
    return loss.mean()


def unet_total_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_coef_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss
