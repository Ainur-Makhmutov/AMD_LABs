import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.conv = DoubleConv(c_in + c_skip, c_out)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:],
                            mode="bilinear", align_corners=False)

        return self.conv(torch.cat([x, skip], dim=1))

class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.e1 = DoubleConv(3, 32)                 # 1/1
        self.e2 = DoubleConv(32, 64)                # 1/2
        self.e3 = DoubleConv(64, 128)               # 1/4
        self.mid = DoubleConv(128, 256)             # 1/8, узкое место
        self.pool = nn.MaxPool2d(2)

        self.u3 = Up(256, 128, 128)
        self.u2 = Up(128, 64, 64)
        self.u1 = Up(64, 32, 32)
        self.out = nn.Conv2d(32, n_classes, 1)   # логиты


    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        x = self.mid(self.pool(s3))
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)

        return self.out(x)


model = UNet()
y = model(torch.randn(2, 3, 255, 257))
assert y.shape == (2, 1, 255, 257)