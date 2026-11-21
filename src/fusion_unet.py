import torch
import torch.nn as nn


class FusionUNet(nn.Module):
    """Small 4-channel U-Net (RGB + SAR) used as a *deployment benchmark network*.

    The weights are randomly initialised: no segmentation task is trained or claimed here.
    It is used only to measure model size, FP32 vs INT8 latency and quantization drift.
    """

    def __init__(self, in_channels=4, out_channels=2):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.enc1, self.pool1 = block(in_channels, 32), nn.MaxPool2d(2)
        self.enc2, self.pool2 = block(32, 64), nn.MaxPool2d(2)
        self.bottleneck = block(64, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, stride=2), block(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, stride=2), block(64, 32)
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
