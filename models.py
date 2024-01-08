import torch
from torch import nn
from utils import DownConv, UpConv, ResizeUpConv, ResBlock


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super(Generator, self).__init__()

        self.down0 = DownConv(in_channels, features, kernel_size=7, stride=1, padding=3, batch=False)
        self.down1 = DownConv(features, features * 2, kernel_size=3, stride=2, padding=1, batch=False)
        self.down2 = DownConv(features * 2, features * 4, kernel_size=3, stride=2, padding=1, batch=False)
        self.down3 = DownConv(features * 4, features * 8, kernel_size=3, stride=2, padding=1, batch=False)

        self.res = ResBlock(features * 8, features * 8, batch=False)

        self.up0 = ResizeUpConv(features * 8, features * 4, kernel_size=3, stride=1, padding=1, batch=False)
        self.up1 = ResizeUpConv(features * 4 * 2, features * 2, kernel_size=3, stride=1, padding=1, batch=False)
        self.up2 = ResizeUpConv(features * 2 * 2, features, kernel_size=3, stride=1, padding=1, batch=False)
        self.up3 = DownConv(features, in_channels, kernel_size=7, stride=1, padding=3, norm=False, batch=False, activation=False)

    def forward(self, x):
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        res = down3.clone()
        for i in range(0, 9):
            res = self.res(res)

        up0 = self.up0(res)
        up1 = self.up1(torch.cat([up0, down2], 1))
        up2 = self.up2(torch.cat([up1, down1], 1))
        up3 = self.up3(up2)

        # use tanh when data is [-1, 1]
        # and sigmoid when data is [0, 1]
        return nn.Sigmoid()(up3)


class Discriminator(nn.Module):
    def __init__(self, in_channels=6, features=32):
        super(Discriminator, self).__init__()
        self.conv0 = DownConv(in_channels, features, kernel_size=4, stride=2, padding=1, norm=False)
        self.conv1 = DownConv(features, features * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = DownConv(features * 2, features * 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = DownConv(features * 4, features * 8, kernel_size=4, stride=1, padding=1)
        self.conv4 = DownConv(features * 8, 1, kernel_size=4, stride=1, padding=1, norm=False, activation=False)

    def forward(self, x, y):
        out = torch.cat([x, y], 1)
        out = self.conv0(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out
