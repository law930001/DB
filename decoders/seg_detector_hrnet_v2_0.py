from collections import OrderedDict
from dwconv import dwconv

import torch
import torch.nn as nn

import numpy as np
import cv2
from concern.visualizer import Visualize

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)


# depthwise_separable_conv
class dsconv(nn.Module):
    def __init__(self, nin, nout, stride, kernels_per_layer=1): 
        super(dsconv, self).__init__() 

        self.operation = nn.Sequential(
            nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, stride=stride, padding=1, groups=nin),
            nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1),
            nn.BatchNorm2d(nin),
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
        self.L1 = dsconv(inner_channels, inner_channels, 1)
        self.L2 = dsconv(inner_channels, inner_channels, 1)
        self.L3 = dsconv(inner_channels, inner_channels, 1)
        self.L4 = dsconv(inner_channels, inner_channels, 1)
        
        # mid layer
        self.E1 = dsconv(inner_channels, inner_channels, 1)
        self.E2 = dsconv(inner_channels, inner_channels, 1)
        self.E3 = dsconv(inner_channels, inner_channels, 1)

        # upsample
        self.U1 = nn.Upsample(size=120, mode='nearest')
        self.U2 = nn.Upsample(size=60, mode='nearest')
        self.U3 = nn.Upsample(size=30, mode='nearest')

        self.UU1 = nn.Upsample(size=160, mode='nearest')
        self.UU2 = nn.Upsample(size=80, mode='nearest')
        self.UU3 = nn.Upsample(size=40, mode='nearest')

        # pooling
        self.D1 = nn.AdaptiveAvgPool2d(120)
        self.D2 = nn.AdaptiveAvgPool2d(60)
        self.D3 = nn.AdaptiveAvgPool2d(30)

        self.DD1 = nn.AdaptiveAvgPool2d(80)
        self.DD2 = nn.AdaptiveAvgPool2d(40)
        self.DD3 = nn.AdaptiveAvgPool2d(20)

    def forward(self, features):

        l1, E1, l2, E2, l3, E3, l4 = features

        l1 = self.L1(l1 + self.UU1(E1))
        l2 = self.L2(self.DD1(E1) + l2 + self.UU2(E2))
        l3 = self.L3(self.DD2(E2) + l3 + self.UU3(E3))
        l4 = self.L4(self.DD3(E3) + l4)

        E1 = self.E1(self.D1(l1) + E1 + self.U1(l2))
        E2 = self.E2(self.D2(l2) + E2 + self.U2(l3))
        E3 = self.E3(self.D3(l3) + E3 + self.U3(l4))

        l1 = l1 + self.UU1(E1)
        l2 = self.DD1(E1) + l2 + self.UU2(E2)
        l3 = self.DD2(E2) + l3 + self.UU3(E3)
        l4 = self.DD3(E3) + l4

        output = l1, E1, l2, E2, l3, E3, l4

        return output


class SegDetector_hrnet48_v2_0(nn.Module):
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
        super(SegDetector_hrnet48_v2_0, self).__init__()
        self.k = k
        self.serial = serial

        # in channel
        self.in1 = nn.Conv2d(in_channels[0], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[1], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[2], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[3], inner_channels, 1, bias=bias)

        self.in1.apply(weights_init)
        self.in2.apply(weights_init)
        self.in3.apply(weights_init)
        self.in4.apply(weights_init)

        # upsample
        self.u_1 = nn.Upsample(size=120, mode='nearest')
        self.u_2 = nn.Upsample(size=60, mode='nearest')
        self.u_3 = nn.Upsample(size=30, mode='nearest')

        # pooling
        self.d_1 = nn.AdaptiveAvgPool2d(120)
        self.d_2 = nn.AdaptiveAvgPool2d(60)
        self.d_3 = nn.AdaptiveAvgPool2d(30)

        # expansion layer
        self.E1 = dsconv(inner_channels, inner_channels, 1)
        self.E2 = dsconv(inner_channels, inner_channels, 1)
        self.E3 = dsconv(inner_channels, inner_channels, 1)

        # FPN
        self.R = 1
        self.fpn_layer = []
        for i in range(0, self.R):
            self.fpn_layer.append(FPN_layer(inner_channels=inner_channels))
        self.fpn = nn.Sequential(*self.fpn_layer)


        # out
        self.out7 = nn.Sequential(
            nn.Conv2d(inner_channels, 256 - (inner_channels //
                      7 * 6), 3, padding=1, bias=bias),
            nn.Upsample(size=160, mode='nearest'))
        self.out6 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      7, 3, padding=1, bias=bias),
            nn.Upsample(size=160, mode='nearest'))
        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      7, 3, padding=1, bias=bias),
            nn.Upsample(size=160, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      7, 3, padding=1, bias=bias),
            nn.Upsample(size=160, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      7, 3, padding=1, bias=bias),
            nn.Upsample(size=160, mode='nearest'))
        self.out2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      7, 3, padding=1, bias=bias),
            nn.Upsample(size=160, mode='nearest'))
        self.out1 = nn.Conv2d(
            inner_channels, inner_channels//7, 3, padding=1, bias=bias)

        self.out1.apply(weights_init)
        self.out2.apply(weights_init)
        self.out3.apply(weights_init)
        self.out4.apply(weights_init)
        self.out5.apply(weights_init)
        self.out6.apply(weights_init)
        self.out7.apply(weights_init)

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

        E1 = self.E1(self.d_1(l1) + self.u_1(l2))
        E2 = self.E2(self.d_2(l2) + self.u_2(l3))
        E3 = self.E3(self.d_3(l3) + self.u_3(l4))

        input_features = l1, E1, l2, E2, l3, E3, l4
        l1, E1, l2, E2, l3, E3, l4 = self.fpn(input_features)

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
