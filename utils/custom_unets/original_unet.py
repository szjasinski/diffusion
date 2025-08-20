import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # padding is 0 in the original paper
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)

        return skip, down


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        # in original paper 
        # the sizes went from 572^2 to 32^2 (4 downsamplings, padding = 0)
        # out channels went from 64 to 512
        # output segmentation map has smaller size than input image
        super().__init__()
        self.downsample_1 = DownSample(in_channels, 64)
        self.downsample_2 = DownSample(64, 128)
        self.downsample_3 = DownSample(128, 256)
        self.downsample_4 = DownSample(256, 512)

        self.bottle_neck = ConvBlock(512, 1024)

        self.upsample_1 = UpSample(1024, 512)
        self.upsample_2 = UpSample(512, 256)
        self.upsample_3 = UpSample(256, 128)
        self.upsample_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)


    def forward(self, x):
        skip_1, down_1 = self.downsample_1(x)
        skip_2, down_2 = self.downsample_2(down_1)
        skip_3, down_3 = self.downsample_3(down_2)
        skip_4, down_4 = self.downsample_4(down_3)

        b = self.bottle_neck(down_4)

        up_1 = self.upsample_1(b, skip_4)
        up_2 = self.upsample_2(up_1, skip_3)
        up_3 = self.upsample_3(up_2, skip_2)
        up_4 = self.upsample_4(up_3, skip_1)

        out = self.out(up_4)

        return out
