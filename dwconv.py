import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))

        return x

class dwconv(nn.Module):
    def __init__(self, nin, stride, expand_ratio = 2): 
        super(dwconv, self).__init__() 

        nin = nin // 2

        hidden_dim = nin * expand_ratio

        self.conv = nn.Conv2d(nin, nin, 1, stride=stride)

        self.operation = nn.Sequential(
                # pw
                nn.Conv2d(nin, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Mish(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Mish(),
                # pw-linear
                nn.Conv2d(hidden_dim, nin, 1, 1, 0, bias=False),
                nn.BatchNorm2d(nin),
            )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

    def forward(self, x):

        x0, x1 = self.channel_shuffle(x)

        out = self.operation(x0) 
        out += self.conv(x1)
        
        return out

if __name__ == "__main__":
    x = torch.autograd.Variable(torch.Tensor(1,256,640,640))
    model = dwconv(nin=256, stride=1)
    model(x)
