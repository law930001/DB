from collections import OrderedDict

from torch.nn.functional import pad
from decoders.neck.fpem_v2 import FPEM_v2
from dwconv import dwconv

import torch
import torch.nn as nn

import numpy as np
import cv2
from concern.visualizer import Visualize

# x = 640
# y = 640
x = 2048
y = 1152

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)


class conv(nn.Module):
    def __init__(self, nin, nout, stride, kernels_per_layer=1, bias=False): 
        super(conv, self).__init__() 

        self.operation = nn.Sequential(
            # nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, stride=stride, padding=1, groups=nin),
            # nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1),
            nn.Conv2d(nin, nout, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True)
        )
        
        self.operation.apply(weights_init)

    def forward(self, x): 
        out = self.operation(x) 

        return out

class FPN_layer(nn.Module):
    def __init__(self, inner_channels=256): 
        super(FPN_layer, self).__init__()

        # main Layer
        self.L1 = conv(inner_channels * 2, inner_channels, 1)
        self.L2 = conv(inner_channels * 3, inner_channels, 1)
        self.L3 = conv(inner_channels * 3, inner_channels, 1)
        self.L4 = conv(inner_channels * 2, inner_channels, 1)
        
        # mid layer
        self.E1 = conv(inner_channels * 3, inner_channels, 1)
        self.E2 = conv(inner_channels * 3, inner_channels, 1)
        self.E3 = conv(inner_channels * 3, inner_channels, 1)

        # last layer
        self.F1 = nn.Conv2d(inner_channels * 2, inner_channels, 1, bias=False)
        self.F2 = nn.Conv2d(inner_channels * 3, inner_channels, 1, bias=False)
        self.F3 = nn.Conv2d(inner_channels * 3, inner_channels, 1, bias=False)
        self.F4 = nn.Conv2d(inner_channels * 2, inner_channels, 1, bias=False)


        # upsample
        self.U1 = nn.Upsample(size=(y * 3 // 16, x * 3 // 16), mode='bilinear')
        self.U2 = nn.Upsample(size=(y * 3 // 32, x * 3 // 32), mode='bilinear')
        self.U3 = nn.Upsample(size=(y * 3 // 64, x * 3 // 64), mode='bilinear')

        self.UU1 = nn.Upsample(size=(y // 4, x // 4), mode='bilinear')
        self.UU2 = nn.Upsample(size=(y // 8, x // 8), mode='bilinear')
        self.UU3 = nn.Upsample(size=(y // 16, x // 16), mode='bilinear')

        # pooling
        self.D1 = nn.AdaptiveAvgPool2d((y * 3 // 16, x * 3 // 16))
        self.D2 = nn.AdaptiveAvgPool2d((y * 3 // 32, x * 3 // 32))
        self.D3 = nn.AdaptiveAvgPool2d((y * 3 // 64, x * 3 // 64))

        self.DD1 = nn.AdaptiveAvgPool2d((y // 8, x // 8))
        self.DD2 = nn.AdaptiveAvgPool2d((y // 16, x // 16))
        self.DD3 = nn.AdaptiveAvgPool2d((y // 32, x // 32))

    def forward(self, features):

        l1, E1, l2, E2, l3, E3, l4 = features

        l1 = self.L1(torch.cat((l1, self.UU1(E1)), 1))
        l2 = self.L2(torch.cat((self.DD1(E1), l2, self.UU2(E2)), 1))
        l3 = self.L3(torch.cat((self.DD2(E2), l3, self.UU3(E3)), 1))
        l4 = self.L4(torch.cat((self.DD3(E3), l4), 1))

        E1 = self.E1(torch.cat((self.D1(l1), E1, self.U1(l2)), 1))
        E2 = self.E2(torch.cat((self.D2(l2), E2, self.U2(l3)), 1))
        E3 = self.E3(torch.cat((self.D3(l3), E3, self.U3(l4)), 1))

        l1 = torch.cat((l1, self.UU1(E1)), 1)
        l2 = torch.cat((self.DD1(E1), l2, self.UU2(E2)), 1)
        l3 = torch.cat((self.DD2(E2), l3, self.UU3(E3)), 1)
        l4 = torch.cat((self.DD3(E3), l4), 1)

        l1 = self.F1(l1)
        l2 = self.F2(l2)
        l3 = self.F3(l3)
        l4 = self.F4(l4)


        output = l1, E1, l2, E2, l3, E3, l4

        return output


class SegDetector_efficientb7_v2_2(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector_efficientb7_v2_2, self).__init__()
        self.k = k
        self.serial = serial
        inner_channels = 128

        # in channel
        self.in1 = nn.Conv2d(in_channels[0], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[1], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[2], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[3], inner_channels, 1, bias=bias)

        self.in1.apply(weights_init)
        self.in2.apply(weights_init)
        self.in3.apply(weights_init)
        self.in4.apply(weights_init)

        # expansion layer
        self.E1 = conv(inner_channels * 2, inner_channels, 1)
        self.E2 = conv(inner_channels * 2, inner_channels, 1)
        self.E3 = conv(inner_channels * 2, inner_channels, 1)

        E1_size = (y * 3 // 16, x * 3 // 16)
        E2_size = (y * 3 // 32, x * 3 // 32)
        E3_size = (y * 3 // 64, x * 3 // 64)

        # upsample
        self.u_1 = nn.Upsample(size=E1_size, mode='bilinear')
        self.u_2 = nn.Upsample(size=E2_size, mode='bilinear')
        self.u_3 = nn.Upsample(size=E3_size, mode='bilinear')

        # pooling
        self.d_1 = nn.AdaptiveAvgPool2d(E1_size)
        self.d_2 = nn.AdaptiveAvgPool2d(E2_size)
        self.d_3 = nn.AdaptiveAvgPool2d(E3_size)

        # FPN
        self.R = 1
        self.fpn_layer = []
        for i in range(0, self.R):
            self.fpn_layer.append(FPN_layer(inner_channels=inner_channels))
        self.fpn = nn.Sequential(*self.fpn_layer)

        # last conv1x1

        self.last_conv = nn.Conv2d(inner_channels, inner_channels, 1, bias=False)

        # FPEMv2
        self.fpem = FPEM_v2(in_channels=128, out_channels=128)


        # out
        self.out7 = nn.Sequential(
            nn.Conv2d(inner_channels, 256 - (256 // 7 * 6), 3, padding=1, bias=bias),
            nn.Upsample(size=(y // 4, x // 4), mode='bilinear'))
        self.out6 = nn.Sequential(
            nn.Conv2d(inner_channels, 256 // 7, 3, padding=1, bias=bias),
            nn.Upsample(size=(y // 4, x // 4), mode='bilinear'))
        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, 256 // 7, 3, padding=1, bias=bias),
            nn.Upsample(size=(y // 4, x // 4), mode='bilinear'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, 256 // 7, 3, padding=1, bias=bias),
            nn.Upsample(size=(y // 4, x // 4), mode='bilinear'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, 256 // 7, 3, padding=1, bias=bias),
            nn.Upsample(size=(y // 4, x // 4), mode='bilinear'))
        self.out2 = nn.Sequential(
            nn.Conv2d(inner_channels, 256 // 7, 3, padding=1, bias=bias),
            nn.Upsample(size=(y // 4, x // 4), mode='bilinear'))
        self.out1 = nn.Sequential(
            nn.Conv2d(inner_channels, 256 // 7, 3, padding=1, bias=bias))


        self.out1.apply(weights_init)
        self.out2.apply(weights_init)
        self.out3.apply(weights_init)
        self.out4.apply(weights_init)
        self.out5.apply(weights_init)
        self.out6.apply(weights_init)
        self.out7.apply(weights_init)


        inner_channels = 256
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(weights_init)


    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        l1, l2, l3, l4 = features

        l1 = self.in1(l1)
        l2 = self.in2(l2)
        l3 = self.in3(l3)
        l4 = self.in4(l4)

        E1 = self.E1(torch.cat((self.d_1(l1), self.u_1(l2)), 1))
        E2 = self.E2(torch.cat((self.d_2(l2), self.u_2(l3)), 1))
        E3 = self.E3(torch.cat((self.d_3(l3), self.u_3(l4)), 1))

        input_features = l1, E1, l2, E2, l3, E3, l4
        
        l1, E1, l2, E2, l3, E3, l4 = self.fpn(input_features)

        l1, E1, l2, E2, l3, E3, l4 = self.fpem(l1, E1, l2, E2, l3, E3, l4)

        l1 = self.last_conv(l1)
        E1 = self.last_conv(E1)
        l2 = self.last_conv(l2)
        E2 = self.last_conv(E2)
        l3 = self.last_conv(l3)
        E3 = self.last_conv(E3)
        l4 = self.last_conv(l4)

        p1 = self.out1(l1)
        p2 = self.out2(E1)
        p3 = self.out3(l2)
        p4 = self.out4(E2)
        p5 = self.out5(l3)
        p6 = self.out6(E3)
        p7 = self.out7(l4)

        fuse = torch.cat((p1, p2, p3, p4, p5, p6, p7), 1)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
