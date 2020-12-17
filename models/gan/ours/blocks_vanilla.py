
"""
https://arxiv.org/pdf/1808.03833v3.pdf
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from modules import Conv2dSame, Interpolate, Identity

class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, activation=nn.ELU(True), bn=False):
        super().__init__()

        assert activation is not None

        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels) if bn else Identity()
        self.conv_features = Conv2dSame(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, x):
        return self.activation(self.bn(self.conv_features(x)))

def conv_relu(in_channels, out_channels, kernel_size, stride=1, dilation=1):
    return nn.Sequential(
        Conv2d(in_channels, out_channels, kernel_size, stride, dilation, activation=nn.ELU(True), bn=True)
    )


def deconv_relu(in_channels, out_channels, kernel_size, stride=2, dilation=1):
    return nn.Sequential(
        Interpolate(scale_factor=stride),
        conv_relu(in_channels, out_channels, kernel_size, stride=1, dilation=dilation)
    )

# Yellow
class ResBlock0(nn.Module):

    def __init__(self, in_channels, d1, d2):
        super().__init__()

        self.enter = conv_relu(in_channels, d1, kernel_size=1)
        self.upper = nn.Sequential(
            conv_relu(d1, d1, kernel_size=3),
            Conv2d(d1, d2, kernel_size=1, activation=Identity())
        )
        self.lower = Conv2d(d1, d2, kernel_size=1, activation=Identity())

    def forward(self, x):
        out = self.enter(x)
        out_upper = self.upper(out)
        out_lower = self.lower(out)
        return out_upper + out_lower

# Green
class ResBlock1(nn.Module):

    def __init__(self, in_channels, d1, d2):
        super().__init__()

        assert in_channels == d2

        self.upper = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ELU(True),

            conv_relu(in_channels, d1, kernel_size=1),
            conv_relu(d1, d1, kernel_size=3),

            Conv2d(d1, d2, kernel_size=1, activation=Identity())
        )

    def forward(self, x):
        out_upper = self.upper(x)
        return x + out_upper

# Dark Green
class ResBlock2(nn.Module):

    def __init__(self, in_channels, d1, d2, stride):
        super().__init__()

        self.enter = nn.Sequential(
            nn.ELU(True),
            nn.BatchNorm2d(in_channels)
        )

        self.upper = nn.Sequential(
            conv_relu(in_channels, d1, kernel_size=1),
            conv_relu(d1, d1, kernel_size=3, stride=stride),
            Conv2d(d1, d2, kernel_size=1, activation=Identity())
        )

        self.lower = Conv2d(in_channels, d2, kernel_size=1, stride=stride, activation=Identity())

    def forward(self, x):
        out = self.enter(x)
        out_upper = self.upper(out)
        out_lower = self.lower(out)
        return out_upper + out_lower

# Cyan
class ResBlock3(nn.Module):

    def __init__(self, in_channels, d1, d2, d3, r1, r2):
        super().__init__()

        assert in_channels == d2

        self.upper_upper = conv_relu(d1, d3 // 2, kernel_size=3, dilation=r1)
        self.upper_lower = nn.Sequential(
            Conv2d(d1, d3 //2, kernel_size=3, dilation=r2, activation=Identity()),
            nn.BatchNorm2d(d3 // 2)
        )

        self.upper_enter = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ELU(True),

            conv_relu(in_channels, d1, kernel_size=1)
        )

        self.upper_exit = Conv2d(d3, d2, kernel_size=1, activation=Identity())

    def forward(self, x):
        out_enter = self.upper_enter(x)
        out_upper_upper = self.upper_upper(out_enter)
        out_upper_lower = self.upper_lower(out_enter)
        out_upper = self.upper_exit(
            torch.cat([out_upper_upper, out_upper_lower], dim=1)
        )
        return x + out_upper

# Purple
class ResBlock4(nn.Module):

    def __init__(self, in_channels, d1, d2, d3, r1, r2):
        super().__init__()

        self.enter = nn.Sequential(
            nn.ELU(True),
            nn.BatchNorm2d(in_channels)
        )

        self.upper_enter = conv_relu(in_channels, d1, kernel_size=1)

        self.upper_upper = conv_relu(d1, d3 // 2, kernel_size=3, dilation=r1)
        self.upper_lower = nn.Sequential(
            Conv2d(d1, d3 //2, kernel_size=3, dilation=r2, activation=Identity()),
            nn.BatchNorm2d(d3 // 2)
        )


        self.upper_exit = Conv2d(d3, d2, kernel_size=1, activation=Identity())
        self.lower = Conv2d(in_channels, d2, kernel_size=1, activation=Identity())

    def forward(self, x):
        out = self.enter(x)
        out_upper_enter = self.upper_enter(out)
        out_upper_upper = self.upper_upper(out_upper_enter)
        out_upper_lower = self.upper_lower(out_upper_enter)
        out_upper_exit = self.upper_exit(
            torch.cat([out_upper_upper, out_upper_lower], dim=1)
        )
        out_lower = self.lower(out)
        return out_upper_exit + out_lower

def encoder(in_channels, base_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),

        # NOTE Added by me
        conv_relu(in_channels, base_channels, kernel_size=5, stride=1),

        # conv_7x7_out
        conv_relu(base_channels, base_channels * 2, kernel_size=3, stride=2),
        conv_relu(base_channels * 2, base_channels * 2, kernel_size=3, stride=1),

        # Residual blocks
        ResBlock0(base_channels * 2, d1=base_channels,     d2=base_channels * 2,         ),
        ResBlock1(base_channels * 2, d1=base_channels,     d2=base_channels * 2,         ),
        ResBlock1(base_channels * 2, d1=base_channels,     d2=base_channels * 2,         ),
        ResBlock2(base_channels * 2, d1=base_channels, d2=base_channels * 4, stride=2),
        ResBlock1(base_channels * 4, d1=base_channels, d2=base_channels * 4,         ),
        ResBlock1(base_channels * 4, d1=base_channels, d2=base_channels * 4,         ),
        ResBlock3(base_channels * 4, d1=base_channels, d2=base_channels * 4, d3=base_channels, r1=1, r2=2),

        ResBlock2(base_channels * 4,  d1=base_channels * 2, d2=base_channels * 8, stride=2),
        ResBlock1(base_channels * 8, d1=base_channels * 2, d2=base_channels * 8),

        ResBlock3(base_channels * 8, d1=base_channels * 2, d2=base_channels * 8, d3=base_channels * 4, r1=1, r2=2),
        ResBlock3(base_channels * 8, d1=base_channels * 2, d2=base_channels * 8, d3=base_channels * 4, r1=1, r2=16),
        ResBlock3(base_channels * 8, d1=base_channels * 2, d2=base_channels * 8, d3=base_channels * 4, r1=1, r2=8),
        ResBlock3(base_channels * 8, d1=base_channels * 2, d2=base_channels * 8, d3=base_channels * 4, r1=1, r2=4)
    )

def residuals(in_channels, out_channels):
    return nn.Sequential(
        ResBlock4(in_channels, d1=in_channels // 4, d2=out_channels, d3=in_channels // 4, r1=2, r2=4),
        ResBlock3(out_channels, d1=out_channels // 4, d2=out_channels, d3=out_channels // 4, r1=2, r2=8),
        ResBlock3(out_channels, d1=out_channels // 4, d2=out_channels, d3=out_channels // 4, r1=2, r2=16)
    )

def decoder(in_channels, out_channels, base_channels, activation):
    return nn.Sequential(
        deconv_relu(in_channels, base_channels * 4, kernel_size=3),
        conv_relu(base_channels * 4, base_channels * 4, kernel_size=3),
        deconv_relu(base_channels * 4, base_channels * 2, kernel_size=3),
        conv_relu(base_channels * 2, base_channels * 2, kernel_size=3),
        deconv_relu(base_channels * 2, base_channels, kernel_size=3),
        conv_relu(base_channels, base_channels, kernel_size=3),
        Conv2d(base_channels, out_channels, kernel_size=1, activation=Identity()),
        activation
    )
