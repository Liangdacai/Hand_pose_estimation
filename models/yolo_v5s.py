import argparse
import logging
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from models.common import Conv, SPP, Focus, BottleneckCSP, Concat



class Detect(nn.Module):
    stride = [8,16,32]  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.training = False

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Pf_yolo_v5(nn.Module):
    def __init__(self):
        super(Pf_yolo_v5,self).__init__()
        self.nc = 1
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
                        [116, 90, 156, 198, 373, 326]]
        self.ch =[128, 256, 512]

        self.Focus1 = Focus(c1 = 3,c2 = 32,k = 3)
        self.Conv2 = Conv(c1 = 32,c2 = 64,k = 3,s = 2)
        self.BottleneckCSP3 = BottleneckCSP(c1 = 64,c2 = 64,n=1)
        self.Conv4 = Conv(c1 = 64,c2 = 128,k = 3,s = 2)
        self.BottleneckCSP5 = BottleneckCSP(c1 = 128, c2 = 128, n = 3)
        self.Conv6 = Conv(c1=128, c2=256, k=3, s=2)
        self.BottleneckCSP7 = BottleneckCSP(c1=256, c2=256, n=3)
        self.Conv8 = Conv(c1=256, c2=512, k=3, s=2)
        self.Spp9 = SPP(c1 = 512,c2 = 512,k = [5,9,13])
        self.BottleneckCSP10 = BottleneckCSP(c1=512, c2=512, n=1,shortcut=False)
        self.Conv11 = Conv(c1=512, c2=256, k=1, s=1)
        self.Upsample12 = nn.modules.upsampling.Upsample(scale_factor=2, mode='nearest')
        self.Concat13 = Concat(1)
        self.BottleneckCSP14 = BottleneckCSP(c1=512, c2=256, n=1, shortcut=False)
        self.Conv15 = Conv(c1=256, c2=128, k=1, s=1)
        self.Upsample16 = nn.modules.upsampling.Upsample(scale_factor=2, mode='nearest')
        self.Concat17 = Concat(1)
        self.BottleneckCSP18 = BottleneckCSP(c1=256, c2=128, n=1, shortcut=False)
        self.Conv19 = Conv(c1=128, c2=128, k=3, s=2)
        self.Concat20 = Concat(1)
        self.BottleneckCSP21 = BottleneckCSP(c1=256, c2=256, n=1, shortcut=False)
        self.Conv22 = Conv(c1=256, c2=256, k=3, s=2)
        self.Concat23 = Concat(1)
        self.BottleneckCSP24 = BottleneckCSP(c1=512, c2=512, n=1, shortcut=False)
        self.Detect = Detect(self.nc,self.anchors,self.ch)



    def forward(self, x):

        # backbone
        x0 = self.Focus1(x)
        x1 = self.Conv2(x0)
        x2 = self.BottleneckCSP3(x1)
        x3 = self.Conv4(x2)
        x4 = self.BottleneckCSP5(x3)
        x5 = self.Conv6(x4)
        x6 = self.BottleneckCSP7(x5)
        x7 = self.Conv8(x6)
        x8 = self.Spp9(x7)
        x9 = self.BottleneckCSP10(x8)
        #head
        x10 = self.Conv11(x9)
        x11 = self.Upsample12(x10)
        x12 = self.Concat13([x11,x6])
        x13 = self.BottleneckCSP14(x12)
        x14 = self.Conv15(x13)
        x15 = self.Upsample16(x14)
        x16 = self.Concat17([x15,x4])
        x17 = self.BottleneckCSP18(x16)
        x18 = self.Conv19(x17)
        x19 = self.Concat20([x18,x14])
        x20 = self.BottleneckCSP21(x19)
        x21 = self.Conv22(x20)
        x22 = self.Concat23([x21,x10])
        x23 = self.BottleneckCSP24(x22)
        #detection
        dete = self.Detect([x17,x20,x23])

        return dete

