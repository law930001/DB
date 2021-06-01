##group conv
from collections import OrderedDict

import torch
import torch.nn as nn
BatchNorm2d = nn.BatchNorm2d

class SegDetector_res152(nn.Module):
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
        super(SegDetector_res152, self).__init__()
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upd3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upd4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upd5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upa3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upa4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upa5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.updd3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.updd4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.updd5 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)
        
        self.down3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels , 3, 
                        padding=1,stride=2 ,bias=bias,groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=1),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        
        self.down4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels , 3, 
                        padding=1,stride=2 ,bias=bias,groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=1),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))

        self.down5 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels , 3, 
            padding=1,stride=2 ,bias=bias,groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=1),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        
        self.downd3 = nn.Sequential(
            nn.Conv2d(64, 64 , 3, 
                        padding=1,stride=2 ,bias=bias,groups=64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        self.downd4 = nn.Sequential(
            nn.Conv2d(64, 64 , 3, 
                        padding=1,stride=2 ,bias=bias,groups=64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='nearest'),
            BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.downd5 = nn.Sequential(
            nn.Conv2d(64, 64 , 3, 
            padding=1,stride=2 ,bias=bias,groups=64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='nearest'),                                    
            BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.out5 = nn.Sequential( 
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias,groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=1),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias,groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=1),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels , 3, padding=1, bias=bias,groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=1),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        
        self.out2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias,groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=1),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))

        self.outd5 = nn.Sequential( 
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias,groups=inner_channels//4),
            nn.Conv2d(inner_channels//4, inner_channels //
                      4, kernel_size=1),
            #nn.Upsample(scale_factor=8, mode='nearest'),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True))
        
        self.outd4 = nn.Sequential( 
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias,groups=inner_channels//4),
            nn.Conv2d(inner_channels//4, inner_channels //
                      4, kernel_size=1),
            #nn.Upsample(scale_factor=4, mode='nearest'),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True))
        
        self.outd3 = nn.Sequential( 
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias,groups=inner_channels//4),
            nn.Conv2d(inner_channels//4, inner_channels //
                      4, kernel_size=1),
            #nn.Upsample(scale_factor=2, mode='nearest'),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True))
        
        self.outd2 = nn.Sequential( 
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias,groups=inner_channels//4),
            nn.Conv2d(inner_channels//4, inner_channels //
                      4, kernel_size=1),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True))


        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        
        self.down3.apply(self.weights_init)
        self.down4.apply(self.weights_init)
        self.down5.apply(self.weights_init)
        
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)
       
        self.outd5.apply(self.weights_init)
        self.outd4.apply(self.weights_init)
        self.outd3.apply(self.weights_init)
        self.outd2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
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
        c2, c3, c4, c5 = features
        #print("c2, c3, c4, c5------shape",c2.shape, c3.shape, c4.shape, c5.shape) 
        # #c2, c3, c4, c5------shape torch.Size([8, 256, 160, 160]) torch.Size([8, 512, 80, 80]) torch.Size([8, 1024, 40, 40]) torch.Size([8, 2048, 20, 20])
        ###########up
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4
        #print("out4~2",out4.shape,out3.shape,out2.shape)
        #out4~2 torch.Size([8, 256, 40, 40]) torch.Size([8, 256, 80, 80]) torch.Size([8, 256, 160, 160])
        p5 = self.out5(in5) #1/32
        p4 = self.out4(out4) #1/16
        p3 = self.out3(out3) #1/8
        p2 = self.out2(out2) #1/4
        #print("p5~2",p5.shape,p4.shape,p3.shape,p2.shape)
        #p5~2 torch.Size([8, 256, 20, 20]) torch.Size([8, 256, 40, 40]) torch.Size([8, 256, 80, 80]) torch.Size([8, 256, 160, 160])

        #########down
        d3 = self.upd3(p3) + p2 #1/4
        d3 = self.down3(d3)
        
        d4 = self.upd4(p4) + d3 #1/8
        d4 = self.down4(d4)
        
        d5 = self.upd5(p5) + d4 #1/16
        d5 = self.down5(d5)
        #print("d3,d4,d5=========",d3.shape,d4.shape,d5.shape)

        ###########up
        a4 = self.upa5(d5) + d4  
        a3 = self.upa4(a4) + d3
        a2 = self.upa3(a3) + p2

        o5 = self.outd5(d5) #1/32
        o4 = self.outd4(a4) #1/16
        o3 = self.outd3(a3) #1/8
        o2 = self.outd2(a2) #1/4
        #print("shape=============",o5.shape,o4.shape,o3.shape,o2.shape) #shape============= torch.Size([1, 64, 184, 320]) torch.Size([1, 64, 184, 320]) torch.Size([1, 64, 184, 320]) torch.Size([1, 64, 184, 320])
        #torch.Size([1, 64, 23, 40]) torch.Size([1, 64, 46, 80]) torch.Size([1, 64, 92, 160]) torch.Size([1, 64, 184, 320])
        
         #########down
        dd3 = self.updd3(o3) + o2 #1/4
        dd3 = self.downd3(dd3)
        
        dd4 = self.updd4(o4) + o3 #1/8
        dd4 = self.downd4(dd4)
        
        dd5 = self.updd5(o5) + o4 #1/16
        dd5 = self.downd5(dd5)

        fuse = torch.cat((dd5, dd4, dd3, o2), 1)
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


#########original
# from collections import OrderedDict

# import torch
# import torch.nn as nn
# BatchNorm2d = nn.BatchNorm2d

# class SegDetector(nn.Module):
#     def __init__(self,
#                  in_channels=[64, 128, 256, 512],
#                  inner_channels=256, k=10,
#                  bias=False, adaptive=False, smooth=False, serial=False,
#                  *args, **kwargs):
#         '''
#         bias: Whether conv layers have bias or not.
#         adaptive: Whether to use adaptive threshold training or not.
#         smooth: If true, use bilinear instead of deconv.
#         serial: If true, thresh prediction will combine segmentation result as input.
#         '''
#         super(SegDetector, self).__init__()
#         self.k = k
#         self.serial = serial
#         self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

#         self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
#         self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
#         self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
#         self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

#         self.out5 = nn.Sequential(
#             nn.Conv2d(inner_channels, inner_channels //
#                       4, 3, padding=1, bias=bias),
#             nn.Upsample(scale_factor=8, mode='nearest'))
#         self.out4 = nn.Sequential(
#             nn.Conv2d(inner_channels, inner_channels //
#                       4, 3, padding=1, bias=bias),
#             nn.Upsample(scale_factor=4, mode='nearest'))
#         self.out3 = nn.Sequential(
#             nn.Conv2d(inner_channels, inner_channels //
#                       4, 3, padding=1, bias=bias),
#             nn.Upsample(scale_factor=2, mode='nearest'))
#         self.out2 = nn.Conv2d(
#             inner_channels, inner_channels//4, 3, padding=1, bias=bias)

#         self.binarize = nn.Sequential(
#             nn.Conv2d(inner_channels, inner_channels //
#                       4, 3, padding=1, bias=bias),
#             BatchNorm2d(inner_channels//4),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
#             BatchNorm2d(inner_channels//4),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
#             nn.Sigmoid())
#         self.binarize.apply(self.weights_init)

#         self.adaptive = adaptive
#         if adaptive:
#             self.thresh = self._init_thresh(
#                     inner_channels, serial=serial, smooth=smooth, bias=bias)
#             self.thresh.apply(self.weights_init)

#         self.in5.apply(self.weights_init)
#         self.in4.apply(self.weights_init)
#         self.in3.apply(self.weights_init)
#         self.in2.apply(self.weights_init)
#         self.out5.apply(self.weights_init)
#         self.out4.apply(self.weights_init)
#         self.out3.apply(self.weights_init)
#         self.out2.apply(self.weights_init)

#     def weights_init(self, m):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             nn.init.kaiming_normal_(m.weight.data)
#         elif classname.find('BatchNorm') != -1:
#             m.weight.data.fill_(1.)
#             m.bias.data.fill_(1e-4)

#     def _init_thresh(self, inner_channels,
#                      serial=False, smooth=False, bias=False):
#         in_channels = inner_channels
#         if serial:
#             in_channels += 1
#         self.thresh = nn.Sequential(
#             nn.Conv2d(in_channels, inner_channels //
#                       4, 3, padding=1, bias=bias),
#             BatchNorm2d(inner_channels//4),
#             nn.ReLU(inplace=True),
#             self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
#             BatchNorm2d(inner_channels//4),
#             nn.ReLU(inplace=True),
#             self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
#             nn.Sigmoid())
#         return self.thresh

#     def _init_upsample(self,
#                        in_channels, out_channels,
#                        smooth=False, bias=False):
#         if smooth:
#             inter_out_channels = out_channels
#             if out_channels == 1:
#                 inter_out_channels = in_channels
#             module_list = [
#                     nn.Upsample(scale_factor=2, mode='nearest'),
#                     nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
#             if out_channels == 1:
#                 module_list.append(
#                     nn.Conv2d(in_channels, out_channels,
#                               kernel_size=1, stride=1, padding=1, bias=True))

#             return nn.Sequential(module_list)
#         else:
#             return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

#     def forward(self, features, gt=None, masks=None, training=False):
#         c2, c3, c4, c5 = features
#         in5 = self.in5(c5)
#         in4 = self.in4(c4)
#         in3 = self.in3(c3)
#         in2 = self.in2(c2)

#         out4 = self.up5(in5) + in4  # 1/16
#         out3 = self.up4(out4) + in3  # 1/8
#         out2 = self.up3(out3) + in2  # 1/4

#         p5 = self.out5(in5)
#         p4 = self.out4(out4)
#         p3 = self.out3(out3)
#         p2 = self.out2(out2)

#         fuse = torch.cat((p5, p4, p3, p2), 1)
#         # this is the pred module, not binarization module; 
#         # We do not correct the name due to the trained model.
#         binary = self.binarize(fuse)
#         if self.training:
#             result = OrderedDict(binary=binary)
#         else:
#             return binary
#         if self.adaptive and self.training:
#             if self.serial:
#                 fuse = torch.cat(
#                         (fuse, nn.functional.interpolate(
#                             binary, fuse.shape[2:])), 1)
#             thresh = self.thresh(fuse)
#             thresh_binary = self.step_function(binary, thresh)
#             result.update(thresh=thresh, thresh_binary=thresh_binary)
#         return result

#     def step_function(self, x, y):
#         return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
