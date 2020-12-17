

import torch
import torch.nn as nn

from modules import Clamp, Identity

from .blocks_vanilla import (
    encoder,
    residuals,
    decoder
)

class Inpainting(nn.Module):

    def __init__(self, input_channels, base_channels=32, range_rgb=[-1, 1]):
        super().__init__()

        self.encoder_current = encoder(input_channels, base_channels) # NOTE +1 is because of the mask

        # Residual blocks
        self.residuals = residuals(base_channels * 8, base_channels * 8)

        # Decoders
        self.decoder_rgb = decoder(base_channels * 8, 3, base_channels, activation=Clamp(*range_rgb))

    def forward(self, current_img, current_mask):
        image_incomplete = current_img * (1 - current_mask)

        # Encode
        out_current = self.encoder_current(torch.cat([image_incomplete, current_mask], dim=1))

        # Residual blocks
        out = self.residuals(out_current)

        # Decode
        rgb = self.decoder_rgb(out)

        return rgb * current_mask + image_incomplete
