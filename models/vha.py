# -*- coding: utf-8 -*-
# ---------------------

from typing import *

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from models import BaseModel


class Upsample(nn.Module):
    """
    Upsamples a given tensor by (scale_factor)X.
    """


    def __init__(self, scale_factor=2, mode='trilinear'):
        # type: (int, str) -> Upsample
        """
        :param scale_factor: the multiplier for the image height / width
        :param mode: the upsampling algorithm - values in {'nearest', 'linear', 'bilinear', 'trilinear'}
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


    def extra_repr(self):
        return f'scale_factor={self.scale_factor}, mode={self.mode}'


# ---------------------

class PixelNorm(nn.Module):
    """
    Normalize the feature vector in each pixel to unit length.
    """


    def __init__(self):
        super().__init__()


    def forward(self, input):
        # type: (torch.Tensor) -> torch.Tensor
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


# ---------------------

class BasicBlock(nn.Module):

    def __init__(self, in_ch, out_ch, mode, pixel_norm, pad=2):
        # type: (int, int, str, bool, Union[int, Tuple[int, int]]) -> BasicBlock
        """
        :param in_ch: numebr of input channels
        :param out_ch: number of output channels
        :param mode: values in {'++', '--', '=='}
        :param pixel_norm: use pixel normalization?
        :param pad: padding
        """
        super().__init__()

        assert mode in ['++', '--', '==']

        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, padding=pad),
            PixelNorm(), nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, padding=pad),
            PixelNorm(), nn.ReLU(inplace=True),
        )

        if not pixel_norm:
            del self.block[1]
            del self.block[3]

        if mode in ['++', '--']:
            if mode == '++':
                final_block_element = Upsample(scale_factor=2)
            else:
                final_block_element = nn.MaxPool3d(kernel_size=2)
            self.block = nn.Sequential(self.block, final_block_element)


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        out = self.block(x)
        return out


# ---------------------

class Autoencoder(BaseModel):

    def __init__(self, hmap_d):
        # type: (int) -> None
        """
        :param hmap_d: number of input channels
        """
        print(f'model: {__file__}\n')
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=hmap_d, out_channels=hmap_d // 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 2, out_channels=hmap_d // 4, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 4, out_channels=hmap_d // 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
        )

        self.fuser = nn.Sequential(
            nn.Conv3d(in_channels=14, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv3d(in_channels=4, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        # --------------

        self.defuser = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv3d(in_channels=4, out_channels=14, kernel_size=5, padding=2),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=hmap_d // 4, out_channels=hmap_d // 4, kernel_size=5, padding=2),
            Upsample(mode='bilinear'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 4, out_channels=hmap_d // 2, kernel_size=5, padding=2),
            Upsample(mode='bilinear'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 2, out_channels=hmap_d, kernel_size=5, padding=2),
            nn.ReLU(True)
        )


    def encode(self, x):
        # type: (torch.Tensor) -> torch.Tensor

        x = self.encoder(torch.reshape(x, (x.shape[0] * 14, x.shape[2], x.shape[3], x.shape[4])).contiguous())
        x = torch.reshape(x, (x.shape[0] // 14, 14, x.shape[1], x.shape[2], x.shape[3])).contiguous()

        x = self.fuser(x)
        return x


    def decode(self, x):
        x = self.defuser(x)

        x = self.decoder(torch.reshape(x, (x.shape[0] * 14, x.shape[2], x.shape[3], x.shape[4])).contiguous())
        x = torch.reshape(x, (x.shape[0] // 14, 14, x.shape[1], x.shape[2], x.shape[3])).contiguous()
        return x


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.encode(x)
        x = self.decode(x)
        return x


# ---------------------


def main():
    import time
    import numpy as np
    from conf import Conf

    cnf = Conf(exp_name='default')

    model = Autoencoder(hmap_d=cnf.hmap_d).to(cnf.device)
    model.load_w('/home/fabio/PycharmProjects/LoCO/models/weights/vha.pth')

    print(model)
    print(f'* number of parameters: {model.n_param}')

    x = torch.rand((1, 14, cnf.hmap_d, cnf.hmap_h, cnf.hmap_w)).to(cnf.device)

    print('\n--- ENCODING ---')
    y = model.encode(x)
    print(f'* input shape: {tuple(x.shape)}')
    print(f'* output shape: {tuple(y.shape)}')

    print(f'* space savings: {100 - 100 * np.prod(tuple(y.shape)) / np.prod(tuple(x.shape)):.2f}%')
    print(f'* factor: {np.prod(tuple(x.shape)) / np.prod(tuple(y.shape)):.2f}')

    print('\n--- DECODING ---')
    xd = model.decode(y)
    print(f'* input shape: {tuple(y.shape)}')
    print(f'* output shape: {tuple(xd.shape)}')

    print('\n--- FORWARD ---')
    y = model.forward(x)
    print(f'* input shape: {tuple(x.shape)}')
    print(f'* output shape: {tuple(y.shape)}')

if __name__ == '__main__':
    main()
