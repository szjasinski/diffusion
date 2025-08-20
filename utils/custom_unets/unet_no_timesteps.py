import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Add residual connections, group norm, more layers, GELU
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)

        return skip, down


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, C=32, t_emb_dim=128):
        # Add sinusoidal timestep embedding, class embedding, attention, einops (instead of pooling)
        super().__init__()

        # down_channels = ((in_channels, C), (C, 2*C), (2*C, 4*C), (4*C, 8*C))
        # up_channels = ...

        self.downsample_1 = DownBlock(in_channels, C)
        self.downsample_2 = DownBlock(C, 2*C)
        self.downsample_3 = DownBlock(2*C, 4*C)
        self.downsample_4 = DownBlock(4*C, 8*C)

        self.bottle_neck = ConvBlock(8*C, 16*C)

        self.upsample_1 = UpBlock(16*C, 8*C)
        self.upsample_2 = UpBlock(8*C, 4*C)
        self.upsample_3 = UpBlock(4*C, 2*C)
        self.upsample_4 = UpBlock(2*C, C)

        self.out = nn.Conv2d(in_channels=C, out_channels=out_channels, kernel_size=1)


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
