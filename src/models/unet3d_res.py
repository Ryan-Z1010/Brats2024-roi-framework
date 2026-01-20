import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = conv3x3(out_ch, out_ch, stride=1)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = self.act(out + identity)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.rb = ResBlock(out_ch + skip_ch, out_ch, stride=1, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # safety for odd shapes (should not happen with 128^3)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.rb(x)

class ResUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=5, base=32, dropout=0.0):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base*2, base*4, base*8, base*16

        self.stem = ResBlock(in_channels, c1, stride=1, dropout=dropout)

        self.down1 = ResBlock(c1, c2, stride=2, dropout=dropout)
        self.down2 = ResBlock(c2, c3, stride=2, dropout=dropout)
        self.down3 = ResBlock(c3, c4, stride=2, dropout=dropout)
        self.down4 = ResBlock(c4, c5, stride=2, dropout=dropout)

        self.up3 = UpBlock(c5, c4, c4, dropout=dropout)
        self.up2 = UpBlock(c4, c3, c3, dropout=dropout)
        self.up1 = UpBlock(c3, c2, c2, dropout=dropout)
        self.up0 = UpBlock(c2, c1, c1, dropout=dropout)

        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b  = self.down4(s3)

        x = self.up3(b, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)
