""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels  # number of channel in input image (original image)
        self.n_classes = n_classes    # number of segment class
        self.bilinear = bilinear
        
        # input torch size : 3, 512, 512
        self.inc = DoubleConv(n_channels, 64)   # 3, 512, 512 -> 64, 512, 512
        self.down1 = Down(64, 128)              # 64, 512, 512 -> 128, 256, 256
        self.down2 = Down(128, 256)             # 128, 256, 256 -> 256, 128, 128
        self.down3 = Down(256, 512)             # 256, 128, 128 -> 512, 64, 64
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # 512, 64, 64 -> 512, 32, 32 (factor = 2)
        self.up1 = Up(1024, 512, bilinear)      # 256, 64, 64
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
#         print('input',x.shape) 
        x1 = self.inc(x) 
#         print('x1:', x1.shape) 
        x2 = self.down1(x1)
#         print('x2:', x2.shape)
        x3 = self.down2(x2)
#         print('x3:', x3.shape)
        x4 = self.down3(x3)
#         print('x4:', x4.shape)
        x5 = self.down4(x4)
#         print('x5:', x5.shape)
        x = self.up1(x5, x4)
#         print('x54:', x.shape)
        x = self.up2(x, x3)
#         print('x543:',  x.shape)
        x = self.up3(x, x2)
#         print('x5432:',  x.shape)
        x = self.up4(x, x1)
#         print('x54321:',  x.shape)
        logits = self.outc(x)
#         print('output:',logits.shape)
        return torch.sigmoid(logits)
    
    
# input torch.Size([4, 3, 512, 512])
# x1: torch.Size([4, 64, 512, 512])
# x2: torch.Size([4, 128, 256, 256])
# x3: torch.Size([4, 256, 128, 128])
# x4: torch.Size([4, 512, 64, 64])
# x5: torch.Size([4, 512, 32, 32])
# x54: torch.Size([4, 256, 64, 64])
# x543: torch.Size([4, 128, 128, 128])
# x5432: torch.Size([4, 64, 256, 256])
# x54321: torch.Size([4, 64, 512, 512])
# output: torch.Size([4, 2, 512, 512])