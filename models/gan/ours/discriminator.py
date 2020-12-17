
from functools import partial

import torch
import torch.nn as nn

from modules import Identity

from .blocks import Conv2d

class Discriminator(nn.Module):

    def __init__(self, in_channels=4, num_channels=64):
        super().__init__()

        global Conv2d
        Conv2d = partial(Conv2d, gated=False, spectral_normalization=True, activation=nn.LeakyReLU(True))

        self.discriminator = nn.Sequential(
            Conv2d(in_channels, num_channels, kernel_size=5, stride=1, ), # conv1
            Conv2d(num_channels, 2 * num_channels, kernel_size=5, stride=2), # conv2
            Conv2d(2 * num_channels, 4 * num_channels, kernel_size=5, stride=2), # conv3
            Conv2d(4 * num_channels, 4 * num_channels, kernel_size=5, stride=2), # conv4
            Conv2d(4 * num_channels, 4 * num_channels, kernel_size=5, stride=2), # conv5
            Conv2d(4 * num_channels, 4 * num_channels, kernel_size=5, stride=2) # conv6
        )

    def forward(self, x):
        return self.discriminator(x)
