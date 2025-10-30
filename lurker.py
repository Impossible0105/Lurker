import numbers
import torch.nn as nn
import torch.nn.functional as F

class LurkerUp(nn.Module):
    def __init__(self, in_channels, scale=2, k_en=1, k_up=5):
        super(LurkerUp, self).__init__()
        assert(isinstance(scale, numbers.Integral))
        self.encoder_conv = nn.Conv2d(in_channels, k_up**2, k_en)
        self.up_k = k_up
        self.unfold = nn.Unfold(kernel_size=self.up_k, stride=1, padding=self.up_k//2)
        self.scale = scale
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.encoder_conv.weight, std=0.001)
    def forward(self, x):
        B, C, H, W = x.shape
        k = self.up_k
        nH = self.scale*H
        nW = self.scale*W
        kernel = self.encoder_conv(x)
        kernel = F.softmax(kernel, dim=1)
        kernel = F.interpolate(kernel, size=(nH,nW), mode='bilinear')
        x = self.unfold(x).view(B, -1, H, W)
        x = F.interpolate(x, size=(nH,nW), mode='nearest')
        x = x.view(B, C, k*k, nH, nW)
        kernel = kernel.unsqueeze(1)
        x = x * kernel
        x = x.sum(dim=2)
        return x


class LurkerDown(nn.Module):
    def __init__(self, in_channels,scale=2,k_en=1, k_down=3):
        super(LurkerDown, self).__init__()
        self.encoder_conv = nn.Conv2d(in_channels, k_down**2, k_en)
        self.down_k = k_down
        self.unfold = nn.Unfold(kernel_size=self.down_k, stride=scale, padding=self.down_k//2)
        self.scale = scale
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.encoder_conv.weight, std=0.001)
    def forward(self, x):
        B, C, H, W = x.shape
        k = self.down_k
        nH = H //self.scale
        nW = W //self.scale
        kernel = self.encoder_conv(x)
        x = self.unfold(x).view(B, C, k*k, H // self.scale, W // self.scale)
        kernel=F.interpolate(kernel, size=(nH,nW), mode='bilinear')
        kernel = F.softmax(kernel, dim=1)
        kernel = kernel.unsqueeze(1)
        x = x * kernel
        x = x.sum(dim=2)
        return x