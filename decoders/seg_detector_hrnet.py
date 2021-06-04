from collections import OrderedDict

import torch
import torch.nn as nn

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

class SegDetector_hrnet48(nn.Module):
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
        super(SegDetector_hrnet48, self).__init__()
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
        self.up_2 = nn.Upsample(scale_factor=2, mode='nearest')

        # downsample

        self.d1 = dsconv(inner_channels, inner_channels, 2)
        self.d2 = dsconv(inner_channels, inner_channels, 2)
        self.d3 = dsconv(inner_channels, inner_channels, 2)

        # block 1
        self.l1_u1 = dsconv(inner_channels, inner_channels, 1)
        self.l1_u2 = dsconv(inner_channels, inner_channels, 2)
        self.l1_u3 = dsconv(inner_channels, inner_channels, 4)
        self.l1_u4 = dsconv(inner_channels, inner_channels, 8)

        self.l2_u1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.l2_u2 = dsconv(inner_channels, inner_channels, 1)
        self.l2_u3 = dsconv(inner_channels, inner_channels, 2)
        self.l2_u4 = dsconv(inner_channels, inner_channels, 4)

        self.l3_u1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.l3_u2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.l3_u3 = dsconv(inner_channels, inner_channels, 1)
        self.l3_u4 = dsconv(inner_channels, inner_channels, 2)

        self.l4_u1 = nn.Upsample(scale_factor=8, mode='nearest')
        self.l4_u2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.l4_u3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.l4_u4 = dsconv(inner_channels, inner_channels, 1)

        # block2
        self.u1_d1 = dsconv(inner_channels, inner_channels, 1)
        self.u1_d2 = dsconv(inner_channels, inner_channels, 2)
        self.u1_d3 = dsconv(inner_channels, inner_channels, 4)
        self.u1_d4 = dsconv(inner_channels, inner_channels, 8)

        self.u2_d1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.u2_d2 = dsconv(inner_channels, inner_channels, 1)
        self.u2_d3 = dsconv(inner_channels, inner_channels, 2)
        self.u2_d4 = dsconv(inner_channels, inner_channels, 4)

        self.u3_d1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.u3_d2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.u3_d3 = dsconv(inner_channels, inner_channels, 1)
        self.u3_d4 = dsconv(inner_channels, inner_channels, 2)

        self.u4_d1 = nn.Upsample(scale_factor=8, mode='nearest')
        self.u4_d2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.u4_d3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.u4_d4 = dsconv(inner_channels, inner_channels, 1)

        # block3
        self.d1_f1 = dsconv(inner_channels, inner_channels, 1)
        self.d1_f2 = dsconv(inner_channels, inner_channels, 2)
        self.d1_f3 = dsconv(inner_channels, inner_channels, 4)
        self.d1_f4 = dsconv(inner_channels, inner_channels, 8)

        self.d2_f1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d2_f2 = dsconv(inner_channels, inner_channels, 1)
        self.d2_f3 = dsconv(inner_channels, inner_channels, 2)
        self.d2_f4 = dsconv(inner_channels, inner_channels, 4)

        self.d3_f1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.d3_f2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d3_f3 = dsconv(inner_channels, inner_channels, 1)
        self.d3_f4 = dsconv(inner_channels, inner_channels, 2)

        self.d4_f1 = nn.Upsample(scale_factor=8, mode='nearest')
        self.d4_f2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.d4_f3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d4_f4 = dsconv(inner_channels, inner_channels, 1)

        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out1 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)

        self.out1.apply(weights_init)
        self.out2.apply(weights_init)
        self.out3.apply(weights_init)
        self.out4.apply(weights_init)

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

        u4 = self.l1_u4(l1) + self.l2_u4(l2) + self.l3_u4(l3) + self.l4_u4(l4)
        u3 = self.l1_u3(l1) + self.l2_u3(l2) + self.l3_u3(l3) + self.l4_u3(l4) + self.up_2(u4)
        u2 = self.l1_u2(l1) + self.l2_u2(l2) + self.l3_u2(l3) + self.l4_u2(l4) + self.up_2(u3)
        u1 = self.l1_u1(l1) + self.l2_u1(l2) + self.l3_u1(l3) + self.l4_u1(l4) + self.up_2(u2)

        d1 = self.u1_d1(u1) + self.u2_d1(u2) + self.u3_d1(u3) + self.u4_d1(u4)
        d2 = self.u1_d2(u1) + self.u2_d2(u2) + self.u3_d2(u3) + self.u4_d2(u4) + self.d1(d1)
        d3 = self.u1_d3(u1) + self.u2_d3(u2) + self.u3_d3(u3) + self.u4_d3(u4) + self.d2(d2)
        d4 = self.u1_d4(u1) + self.u2_d4(u2) + self.u3_d4(u3) + self.u4_d4(u4) + self.d3(d3)

        f4 = self.d1_f4(d1) + self.d2_f4(d2) + self.d3_f4(d3) + self.d4_f4(d4)
        f3 = self.d1_f3(d1) + self.d2_f3(d2) + self.d3_f3(d3) + self.d4_f3(d4) + self.up_2(f4)
        f2 = self.d1_f2(d1) + self.d2_f2(d2) + self.d3_f2(d3) + self.d4_f2(d4) + self.up_2(f3)
        f1 = self.d1_f1(d1) + self.d2_f1(d2) + self.d3_f1(d3) + self.d4_f1(d4) + self.up_2(f2)

        p1 = self.out1(f1)
        p2 = self.out2(f2)
        p3 = self.out3(f3)
        p4 = self.out4(f4)


        fuse = torch.cat((p1, p2, p3, p4), 1)
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
