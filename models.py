import torch
from torch import nn
from utils import DownConv, UpConv, ResizeUpConv, ResBlock


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super().__init__()

        self.down0 = DownConv(in_channels, features, kernel_size=4, stride=2, padding=1, norm=False, batch=False)
        self.down1 = DownConv(features, features * 2, kernel_size=4, stride=2, padding=1, batch=False)
        self.down2 = DownConv(features * 2, features * 4, kernel_size=4, stride=2, padding=1, batch=False)
        self.down3 = DownConv(features * 4, features * 8, kernel_size=4, stride=2, padding=1, batch=False)
        self.down4 = DownConv(features * 8, features * 8, kernel_size=4, stride=4, padding=0, batch=False)
        self.down5 = DownConv(features * 8, features * 8, kernel_size=4, stride=2, padding=1, norm=False, batch=False, activation=True, leaky=False)

        # not using dropout in up0
        self.up0 = UpConv(features * 8, features * 8, kernel_size=4, stride=2, padding=1, batch=False)
        self.up1 = UpConv(features * 8 * 2, features * 8, kernel_size=4, stride=4, padding=0, batch=False)
        self.up2 = UpConv(features * 8 * 2, features * 4, kernel_size=4, stride=2, padding=1, batch=False)
        self.up3 = UpConv(features * 4 * 2, features * 2, kernel_size=4, stride=2, padding=1, batch=False)
        self.up4 = UpConv(features * 2 * 2, features, kernel_size=4, stride=2, padding=1, batch=False)
        self.up5 = UpConv(features * 2, in_channels, kernel_size=4, stride=2, padding=1, norm=False, batch=False, activation=False)

    def forward(self, x):
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)

        up0 = self.up0(down5)
        up1 = self.up1(torch.cat([up0, down4], 1))
        up2 = self.up2(torch.cat([up1, down3], 1))
        up3 = self.up3(torch.cat([up2, down2], 1))
        up4 = self.up4(torch.cat([up3, down1], 1))
        up5 = self.up5(torch.cat([up4, down0], 1))

        # use tanh when data is [-1, 1]
        # and sigmoid when data is [0, 1]
        return nn.Sigmoid()(up5)


# class Generator(nn.Module):
#     def __init__(self, in_channels=3, features=32):
#         super(Generator, self).__init__()

#         self.down0 = DownConv(in_channels, features, kernel_size=4, stride=2, padding=1, batch=False, norm=False)
#         self.down1 = DownConv(features, features * 2, kernel_size=4, stride=2, padding=1, batch=False)
#         self.down2 = DownConv(features * 2, features * 4, kernel_size=4, stride=2, padding=1, batch=False)
#         self.down3 = DownConv(features * 4, features * 8, kernel_size=4, stride=2, padding=1, batch=False)

#         self.res = ResBlock(features * 8, features * 8, batch=False)

#         self.up0 = UpConv(features * 8, features * 4, kernel_size=4, stride=2, padding=1, batch=False)
#         self.up1 = UpConv(features * 8, features * 2, kernel_size=4, stride=2, padding=1, batch=False)
#         self.up2 = UpConv(features * 4, features, kernel_size=4, stride=2, padding=1, batch=False)
#         self.up3 = UpConv(features * 2, in_channels, kernel_size=4, stride=2, padding=1, norm=False, batch=False, activation=False)

#     def forward(self, x):
#         down0 = self.down0(x)
#         down1 = self.down1(down0)
#         down2 = self.down2(down1)
#         down3 = self.down3(down2)

#         # res = down3.clone()
#         # for i in range(0, 3):
#         res = self.res(down3)

#         up0 = self.up0(res)
#         up1 = self.up1(torch.cat([up0, down2], 1))
#         up2 = self.up2(torch.cat([up1, down1], 1))
#         up3 = self.up3(torch.cat([up2, down0], 1))

#         # use tanh when data is [-1, 1]
#         # and sigmoid when data is [0, 1]
#         return nn.Sigmoid()(up3)


class Discriminator(nn.Module):
    def __init__(self, in_channels=6, features=32):
        super(Discriminator, self).__init__()
        self.conv0 = DownConv(in_channels, features, kernel_size=4, stride=2, padding=1, norm=False, batch=False)
        self.conv1 = DownConv(features, features * 2, kernel_size=4, stride=2, padding=1, batch=False)
        self.conv2 = DownConv(features * 2, features * 4, kernel_size=4, stride=2, padding=1, batch=False)
        self.conv3 = DownConv(features * 4, features * 8, kernel_size=4, stride=1, padding=1, batch=False)
        self.conv4 = DownConv(features * 8, 1, kernel_size=4, stride=1, padding=1, norm=False, batch=False, activation=False)

    def forward(self, x, y):
        out = torch.cat([x, y], 1)
        out = self.conv0(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out
