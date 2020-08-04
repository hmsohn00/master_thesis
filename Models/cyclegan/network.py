import torch.nn as nn
import torch.nn.functional as F
import torch
import GPUtil


# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """
    Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """
    Creates a transpose convolutional layer, with optional batch normalization.
    """
    layers = []
    # append transpose conv layer
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    # optional batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


# residual block class
class ResidualBlock(nn.Module):
    """
    Defines a residual block.
    """

    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)

        self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim,
                                kernel_size=3, stride=1, padding=1, batch_norm=True)

        self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim,
                                kernel_size=3, stride=1, padding=1, batch_norm=True)

    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out_1 = F.relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1)
        return out_2
    

# Alpha residual block class
class AlphaResidualBlock(nn.Module):
    """
    Defines an alpha residual block.
    """

    def __init__(self, conv_dim):
        super(AlphaResidualBlock, self).__init__()
        # conv_dim = number of inputs
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)

        self.alpha = nn.Parameter(torch.Tensor([0]))
        self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim,
                                kernel_size=3, stride=1, padding=1, batch_norm=True)

        self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim,
                                kernel_size=3, stride=1, padding=1, batch_norm=True)

    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out_1 = F.relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1) * self.alpha
        return out_2


# discriminator class
class Discriminator(nn.Module):

    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()
        # Define all convolutional layers

        # Convolutional layers, increasing in depth
        # first layer has no batchnorm
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) 
        self.conv2 = conv(conv_dim, conv_dim * 2, 4) 
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4) 
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4) 
        
        # Classification layer
        self.conv5 = conv(conv_dim * 8, 1, 4, stride=1, batch_norm=False) 
        self.fc = nn.Sequential(
            nn.Linear(1*31*31, 1),      
            nn.Sigmoid(),
        )
    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv5(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CycleGenerator(nn.Module):

    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        # 1. Define the encoder part of the generator
        self.conv1 = conv(3, conv_dim, 4)                   
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)        
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)   

        # 2. Define the resnet part of the generator
        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim * 4))
        self.res_blocks = nn.Sequential(*res_layers)       

        # 3. Define the decoder part of the generator
        self.deconv1 = deconv(conv_dim * 4, conv_dim * 2, 4)  
        self.deconv2 = deconv(conv_dim * 2, conv_dim, 4)        
        # no batch norm on last layer
        self.deconv3 = deconv(conv_dim, 3, 4, batch_norm=False) 

    def forward(self, x):
        """
        Given an image x, returns a transformed image.
        """
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.res_blocks(out)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = torch.tanh(self.deconv3(out))

        return out


def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """
    Builds the generators and discriminators.
    """

    # Instantiate generators
    G_XtoY = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    G_YtoX = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    # Instantiate discriminators
    D_X = Discriminator(conv_dim=d_conv_dim)
    D_Y = Discriminator(conv_dim=d_conv_dim)
    
    device = torch.device(GPUtil.getFirstAvailable()[0] if torch.cuda.is_available() else "cpu")
    print('device=',device)
    G_XtoY.to(device)
    G_YtoX.to(device)
    D_X.to(device)
    D_Y.to(device)

    return G_XtoY, G_YtoX, D_X, D_Y, device


# helper function for printing the model architecture
def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """
    Prints model information for the generators and discriminators.
    """
    print("                     G_XtoY                    ")
    print("-----------------------------------------------")
    print(G_XtoY)
    print()

    print("                     G_YtoX                    ")
    print("-----------------------------------------------")
    print(G_YtoX)
    print()

    print("                      D_X                      ")
    print("-----------------------------------------------")
    print(D_X)
    print()

    print("                      D_Y                      ")
    print("-----------------------------------------------")
    print(D_Y)
    print()
