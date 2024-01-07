from torch import nn


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm=True, batch=True, activation=True, leaky=True):
        super(DownConv, self).__init__()
        self.activation = activation
        self.norm = norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode="reflect")

        if self.norm:
            self.norm_op = nn.BatchNorm2d(out_channels) if batch else nn.InstanceNorm2d(out_channels, affine=False)
        if self.activation:
            self.activation_op = nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm_op(out)
        if self.activation:
            out = self.activation_op(out)

        return out


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, norm=True, batch=True, activation=True, leaky=False, dropout=False):
        super(UpConv, self).__init__()
        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding)

        if self.norm:
            self.norm_op = nn.BatchNorm2d(out_channels) if batch else nn.InstanceNorm2d(out_channels, affine=False)
        if self.activation:
            self.activation_op = nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        if self.dropout:
            self.dropout_op = nn.Dropout2d(0.5)

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm_op(out)
        if self.activation:
            out = self.activation_op(out)
        if self.dropout:
            out = self.dropout_op(out)

        return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch=True, downsample=None):
        super(ResBlock, self).__init__()
        if batch:
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels * self.expansion)
        else:
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=False)
            self.norm2 = nn.InstanceNorm2d(out_channels * self.expansion, affine=False)

        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="reflect")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="reflect")

    def forward(self, x):
        identity = x.clone()
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity

        return out
