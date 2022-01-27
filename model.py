import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super().__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.body(x)
        return out


class BasicBlockSig(nn.Module):
    def __init__(self, in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1 = BasicBlock(channel, channel // reduction, 3, 1, 3, 3)
        self.c2 = BasicBlock(channel, channel // reduction, 3, 1, 5, 5)
        self.c3 = BasicBlock(channel, channel // reduction, 3, 1, 7, 7)
        self.c4 = BasicBlockSig((channel // reduction)*3, channel, 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.r1 = ResidualBlock(in_channels, out_channels)
        self.r2 = ResidualBlock(in_channels*2, out_channels*2)
        self.r3 = ResidualBlock(in_channels*4, out_channels*4)
        self.g = BasicBlock(in_channels*8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        c0 =  x

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)
                
        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)
               
        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = self.ca(g)
        return out


class DRLN(nn.Module):
    def __init__(self, n_feats=64, scale=2):
        super().__init__()

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)

        self.b1 = Block(n_feats, n_feats)
        self.b2 = Block(n_feats, n_feats)
        self.b3 = Block(n_feats, n_feats)
        self.b4 = Block(n_feats, n_feats)
        self.b5 = Block(n_feats, n_feats)
        self.b6 = Block(n_feats, n_feats)
        self.b7 = Block(n_feats, n_feats)
        self.b8 = Block(n_feats, n_feats)
        self.b9 = Block(n_feats, n_feats)
        self.b10 = Block(n_feats, n_feats)
        self.b11 = Block(n_feats, n_feats)
        self.b12 = Block(n_feats, n_feats)
        self.b13 = Block(n_feats, n_feats)
        self.b14 = Block(n_feats, n_feats)
        self.b15 = Block(n_feats, n_feats)
        self.b16 = Block(n_feats, n_feats)
        self.b17 = Block(n_feats, n_feats)
        self.b18 = Block(n_feats, n_feats)
        self.b19 = Block(n_feats, n_feats)
        self.b20 = Block(n_feats, n_feats)

        self.c1 = BasicBlock(n_feats*2, n_feats, 3, 1, 1)
        self.c2 = BasicBlock(n_feats*3, n_feats, 3, 1, 1)
        self.c3 = BasicBlock(n_feats*4, n_feats, 3, 1, 1)
        self.c4 = BasicBlock(n_feats*2, n_feats, 3, 1, 1)
        self.c5 = BasicBlock(n_feats*3, n_feats, 3, 1, 1)
        self.c6 = BasicBlock(n_feats*4, n_feats, 3, 1, 1)
        self.c7 = BasicBlock(n_feats*2, n_feats, 3, 1, 1)
        self.c8 = BasicBlock(n_feats*3, n_feats, 3, 1, 1)
        self.c9 = BasicBlock(n_feats*4, n_feats, 3, 1, 1)
        self.c10 = BasicBlock(n_feats*2, n_feats, 3, 1, 1)
        self.c11 = BasicBlock(n_feats*3, n_feats, 3, 1, 1)
        self.c12 = BasicBlock(n_feats*4, n_feats, 3, 1, 1)
        self.c13 = BasicBlock(n_feats*2, n_feats, 3, 1, 1)
        self.c14 = BasicBlock(n_feats*3, n_feats, 3, 1, 1)
        self.c15 = BasicBlock(n_feats*4, n_feats, 3, 1, 1)
        self.c16 = BasicBlock(n_feats*5, n_feats, 3, 1, 1)
        self.c17 = BasicBlock(n_feats*2, n_feats, 3, 1, 1)
        self.c18 = BasicBlock(n_feats*3, n_feats, 3, 1, 1)
        self.c19 = BasicBlock(n_feats*4, n_feats, 3, 1, 1)
        self.c20 = BasicBlock(n_feats*5, n_feats, 3, 1, 1)

        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers += [nn.Conv2d(n_feats, 4*n_feats, 3, 1, 1), nn.ReLU(inplace=True)]
                layers += [nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [nn.Conv2d(n_feats, 9*n_feats, 3, 1, 1), nn.ReLU(inplace=True)]
            layers += [nn.PixelShuffle(3)]
        self.upsample = nn.Sequential(*layers)

        self.tail = nn.Conv2d(n_feats, 3, 3, 1, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        a1 = o3 + c0

        b4 = self.b4(a1)
        c4 = torch.cat([o3, b4], dim=1)
        o4 = self.c4(c4)

        b5 = self.b5(o4)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.c5(c5)

        b6 = self.b6(o5)
        c6 = torch.cat([c5, b6], dim=1)
        o6 = self.c6(c6)
        a2 = o6 + a1

        b7 = self.b7(a2)
        c7 = torch.cat([o6, b7], dim=1)
        o7 = self.c7(c7)

        b8 = self.b8(o7)
        c8 = torch.cat([c7, b8], dim=1)
        o8 = self.c8(c8)

        b9 = self.b9(o8)
        c9 = torch.cat([c8, b9], dim=1)
        o9 = self.c9(c9)
        a3 = o9 + a2

        b10 = self.b10(a3)
        c10 = torch.cat([o9, b10], dim=1)
        o10 = self.c10(c10)

        b11 = self.b11(o10)
        c11 = torch.cat([c10, b11], dim=1)
        o11 = self.c11(c11)

        b12 = self.b12(o11)
        c12 = torch.cat([c11, b12], dim=1)
        o12 = self.c12(c12)
        a4 = o12 + a3

        b13 = self.b13(a4)
        c13 = torch.cat([o12, b13], dim=1)
        o13 = self.c13(c13)

        b14 = self.b14(o13)
        c14 = torch.cat([c13, b14], dim=1)
        o14 = self.c14(c14)

        b15 = self.b15(o14)
        c15 = torch.cat([c14, b15], dim=1)
        o15 = self.c15(c15)

        b16 = self.b16(o15)
        c16 = torch.cat([c15, b16], dim=1)
        o16 = self.c16(c16)
        a5 = o16 + a4

        b17 = self.b17(a5)
        c17 = torch.cat([o16, b17], dim=1)
        o17 = self.c17(c17)

        b18 = self.b18(o17)
        c18 = torch.cat([c17, b18], dim=1)
        o18 = self.c18(c18)

        b19 = self.b19(o18)
        c19 = torch.cat([c18, b19], dim=1)
        o19 = self.c19(c19)

        b20 = self.b20(o19)
        c20 = torch.cat([c19, b20], dim=1)
        o20 = self.c20(c20)
        a6 = o20 + a5

        b_out = a6 + x
        out = self.upsample(b_out)

        out = self.tail(out)
        f_out = self.add_mean(out)

        return f_out
