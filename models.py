import torch
from torch import nn
from utils import DownConv, UpConv, ResBlock


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super(Generator, self).__init__()

        self.down0 = DownConv(in_channels, features, kernel_size=4, stride=2, padding=1, norm=False)
        self.down1 = DownConv(features, features * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = DownConv(features * 2, features * 4, kernel_size=4, stride=2, padding=1)
        self.down3 = DownConv(features * 4, features * 8, kernel_size=4, stride=2, padding=1)

        self.res = ResBlock(features * 8, features * 8)

        self.up0 = UpConv(features * 8, features * 4, kernel_size=4, stride=2, padding=1)
        self.up1 = UpConv(features * 4 * 2, features * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = UpConv(features * 2 * 2, features, kernel_size=4, stride=2, padding=1)
        self.up3 = UpConv(features, in_channels, kernel_size=4, stride=2, padding=1, norm=False, activation=False)

    def forward(self, x):
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        res = down3.clone()
        for i in range(0, 3):
            res = self.res(res)

        up0 = self.up0(res)
        up1 = self.up1(torch.cat([up0, down2], 1))
        up2 = self.up2(torch.cat([up1, down1], 1))
        up3 = self.up3(up2)

        # use tanh when data is [-1, 1]
        # and sigmoid when data is [0, 1]
        return nn.Sigmoid()(up3)


class GlobalGenerator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super(GlobalGenerator, self).__init__()

        self.down0 = DownConv(in_channels, features, kernel_size=7, stride=1, padding=3, batch=False)
        self.down1 = DownConv(features, features * 2, kernel_size=3, stride=2, padding=1, batch=False)
        self.down2 = DownConv(features * 2, features * 4, kernel_size=3, stride=2, padding=1, batch=False)
        self.down3 = DownConv(features * 4, features * 8, kernel_size=3, stride=2, padding=1, batch=False)

        self.res = ResBlock(features * 8, features * 8, batch=False)

        self.up0 = UpConv(features * 8, features * 4, kernel_size=3, stride=2, padding=1, output_padding=1, batch=False)
        self.up1 = UpConv(features * 4 * 2, features * 2, kernel_size=3, stride=2, padding=1, output_padding=1, batch=False)
        self.up2 = UpConv(features * 2 * 2, features, kernel_size=3, stride=2, padding=1, output_padding=1, batch=False)
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


class VanillaGenerator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super(VanillaGenerator, self).__init__()

        self.down0 = DownConv(in_channels, features, kernel_size=4, stride=2, padding=1, norm=False)
        self.down1 = DownConv(features, features * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = DownConv(features * 2, features * 4, kernel_size=4, stride=2, padding=1)
        self.down3 = DownConv(features * 4, features * 8, kernel_size=4, stride=2, padding=1)
        self.down4 = DownConv(features * 8, features * 8, kernel_size=4, stride=2, padding=1)
        self.down5 = DownConv(features * 8, features * 8, kernel_size=4, stride=2, padding=1)
        self.down6 = DownConv(features * 8, features * 8, kernel_size=4, stride=2, padding=1)
        self.middle = DownConv(features * 8, features * 8, kernel_size=4, stride=2, padding=1, norm=False, activation=True, leaky=False)

        self.up0 = UpConv(features * 8, features * 8, kernel_size=4, stride=2, padding=1, dropout=True)
        self.up1 = UpConv(features * 8 * 2, features * 8, kernel_size=4, stride=2, padding=1, dropout=True)
        self.up2 = UpConv(features * 8 * 2, features * 8, kernel_size=4, stride=2, padding=1, dropout=True)
        self.up3 = UpConv(features * 8 * 2, features * 8, kernel_size=4, stride=2, padding=1)
        self.up4 = UpConv(features * 8 * 2, features * 4, kernel_size=4, stride=2, padding=1)
        self.up5 = UpConv(features * 4 * 2, features * 2, kernel_size=4, stride=2, padding=1)
        self.up6 = UpConv(features * 2 * 2, features, kernel_size=4, stride=2, padding=1)
        self.up7 = UpConv(features * 2, in_channels, kernel_size=4, stride=2, padding=1, norm=False, activation=False)

    def forward(self, x):
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)
        middle = self.middle(down6)

        up0 = self.up0(middle)
        up1 = self.up1(torch.cat([up0, down6], 1))
        up2 = self.up2(torch.cat([up1, down5], 1))
        up3 = self.up3(torch.cat([up2, down4], 1))
        up4 = self.up4(torch.cat([up3, down3], 1))
        up5 = self.up5(torch.cat([up4, down2], 1))
        up6 = self.up6(torch.cat([up5, down1], 1))
        up7 = self.up7(torch.cat([up6, down0], 1))

        # use tanh when data is [-1, 1]
        # and sigmoid when data is [0, 1]
        return nn.Sigmoid()(up7)


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
