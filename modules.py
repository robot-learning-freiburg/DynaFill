
import torch
import torch.nn as nn
import torch.nn.functional as F

from functional import _get_same_conv_padding

class Identity(nn.Module):

    def forward(self, x):
        return x

class Flatten(nn.Module):

    def forward(self, x):
        super().__init__()

        batch_size = x.size(0)
        return x.view(batch_size, -1)

class Interpolate(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super().__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)

class Clamp(nn.Module):

    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

class Conv2dSame(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        padding = 0
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__dilation = dilation

    def forward(self, x):
        _, C, H, W = x.size()
        start_H, end_H = _get_same_conv_padding(H, self.__kernel_size, self.__stride, self.__dilation)
        start_W, end_W = _get_same_conv_padding(W, self.__kernel_size, self.__stride, self.__dilation)
        pad = nn.ZeroPad2d((
            start_W, end_W, # left & right
            start_H, end_W # top & bottom
        ))
        x_padded = pad(x)
        out = super().forward(x_padded)
        if self.__stride == 1:
            assert out.shape[2:3] == x.shape[2:3], (out.shape, x.shape)
        return out

