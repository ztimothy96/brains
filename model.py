import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv3d, ConvTranspose3d, Dropout, MaxPool3d

kern_init = torch.nn.init.kaiming_uniform_


class ContractionBlock(nn.Module):
    def __init__(self, C_in, C_out, p, pool):
        super(ContractionBlock, self).__init__()
        self.conv1 = Conv3d(C_in, C_out, (3, 3, 3), padding='same')
        self.dropout = Dropout(p)
        self.conv2 = Conv3d(C_out, C_out, (3, 3, 3), padding='same')
        self.pool = MaxPool3d((2, 2, 2)) if pool else None
        kern_init(self.conv1.weight, nonlinearity='relu')
        kern_init(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        c = F.relu(self.conv1(x))
        c = self.dropout(c)
        c = F.relu(self.conv2(c))
        p = self.pool(c) if self.pool else None
        return c, p
    

class ExpansionBlock(nn.Module):
    def __init__(self, C_in, C_out, p):
        super(ExpansionBlock, self).__init__()
        self.trans = ConvTranspose3d(C_in, C_out, (2, 2, 2), stride=(2, 2, 2), padding='same')
        self.conv1 = Conv3d(C_out, C_out, (3, 3, 3), padding='same')
        self.dropout = Dropout(p)
        self.conv2 = Conv3d(C_out, C_out, (3, 3, 3), padding='same')
        kern_init(self.conv1.weight, nonlinearity='relu')
        kern_init(self.conv2.weight, nonlinearity='relu')

    def forward(self, x_old, x):
        x = self.trans(x)
        # Pytorch Conv3D: (N, C, D, H, W)
        x = torch.cat((x, x_old), dim=1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        return x
    

class UNet(nn.Module):
    def __init__(self, C, n_classes):
        # Pytorch Conv3D: (N, C, D, H, W)
        super(UNet, self).__init__()
        self.contract1 = ContractionBlock(C, 16, 0.1, pool=True),
        self.contract2 = ContractionBlock(16, 32, 0.1, pool=True),
        self.contract3 = ContractionBlock(32, 64, 0.2, pool=True),
        self.contract4 = ContractionBlock(64, 128, 0.2, pool=True),
        self.contract5 = ContractionBlock(128, 256, 0.3, pool=False)
        self.expand1 = ExpansionBlock(256, 128, 0.2)
        self.expand2 = ExpansionBlock(128, 64, 0.2)
        self.expand3 = ExpansionBlock(64, 32, 0.1)
        self.expand4 = ExpansionBlock(32, 16, 0.1)
        self.conv = Conv3d(16, n_classes, (1, 1, 1))

    def forward(self, x):
        c1, x = self.contract1(x)
        c2, x = self.contract2(x)
        c3, x = self.contract3(x)
        c4, x = self.contract4(x)
        x, _ = self.contract5(x)
        x = self.expand1(c4, x)
        x = self.expand2(c3, x)
        x = self.expand3(c2, x)
        x = self.expand4(c1, x)
        x = F.softmax(self.conv(x))
        return x

if __name__=='__main__':
    model = UNet(1, 4)
    print(model)
