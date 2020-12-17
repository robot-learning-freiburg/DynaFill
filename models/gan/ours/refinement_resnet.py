
from operator import mul
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Clamp, Identity

from .blocks_vanilla import (
    encoder,
    decoder,
    Conv2d
)

class Gate(nn.Module):

    def __init__(self, channels, n_feature_maps):
        super().__init__()
        self.channels = channels
        self.n_feature_maps = n_feature_maps

        conv = partial(Conv2d, activation=nn.ELU(True), bn=True)

        self.conv_mask = nn.Sequential(
            conv(n_feature_maps * channels, channels, kernel_size=3),
            conv(channels, channels, kernel_size=3), # 8B
            conv(channels, channels // 2, kernel_size=3), # 4B
            conv(channels // 2, channels // 4, kernel_size=3), # 2B
            conv(channels // 4, channels // 8, kernel_size=3), # B

            conv(channels // 8, n_feature_maps, kernel_size=1, activation=nn.Softmax(dim=1), bn=False) if n_feature_maps > 2 \
            else conv(channels // 8, 1, kernel_size=1, activation=torch.sigmoid, bn=False)
        )

    def forward(self, *maps):
        assert len(maps) == self.n_feature_maps
        maps_concatenated = torch.cat(maps, dim=1) # [B, N * C, H, W]
        B, _, H, W = maps_concatenated.size()
        masks = self.conv_mask(maps_concatenated) # [B, N, H, W]
        if self.n_feature_maps > 2:
            combined = sum(map(lambda t: mul(*t), zip(maps, torch.chunk(masks, self.n_feature_maps, dim=1))))
        else:
            combined = maps[0] * (1 - masks) + maps[1] * masks
        return combined
 
class TranslationSimple2(nn.Module):

    def __init__(self, base_channels, n_warped_imgs, range_rgb):
        super().__init__()

        self.encoder_current = encoder(4, base_channels)
        self.encoder_warped_t_1 = encoder(3, base_channels)

        self.gate = Gate(8 * base_channels, n_warped_imgs + 1)

        self.decoder = decoder(base_channels * 8, 3, base_channels, activation=Clamp(*range_rgb)) # RGB

    def forward(self, image_coarse, mask, *images_warped):
        out_encoder_current = self.encoder_current(torch.cat([image_coarse, mask], dim=1))
        out_encoder_warped_t_1 = self.encoder_warped_t_1(images_warped[0])
        out = self.gate(out_encoder_warped_t_1, out_encoder_current)
        out = self.decoder(out)
        return out
