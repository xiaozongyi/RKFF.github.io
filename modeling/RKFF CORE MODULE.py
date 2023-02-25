import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _AtrousModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_AtrousModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Waterfall(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(Waterfall, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            #dilations = [ 6, 12, 18, 24]
            dilations = [24, 18, 12,  6]
        elif output_stride == 8:
            dilations = [48, 36, 24, 12]
        else:
            raise NotImplementedError

        self.atrous1 = _AtrousModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.atrous2 = _AtrousModule(2304, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.atrous3 = _AtrousModule(2560, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.atrous4 = _AtrousModule(2816, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.conv2 = nn.Conv2d(256,256,1,bias=False)
        self.conv3 = nn.Conv2d(2304,256,1,bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        #print(x.shape)2048 33 33
        x1 = self.atrous1(x)
        #print(x1.shape)256 33 33
        #cat,conv 
        xx1=torch.cat((x,x1),dim=1)
        #print(xx1.shape)2304 33 33
        x2 = self.atrous2(xx1)
        #print(x2.shape)256 33 33
        #cat,conv  
        xx2=torch.cat((xx1,x2),dim=1)
        #print(xx2.shape)2560 33 33
        x3 = self.atrous3(xx2)
        #print(x3.shape)2560 33 33
        #cat,conv 
        xx3=torch.cat((xx2,x3),dim=1)
        #print(xx3.shape)2816 33 33
        x4 = self.atrous4(xx3)
        #print(x4.shape)256 33 33

        x1 = self.conv2(x1)
        #print(x1.shape)256 33 33
        x2 = self.conv2(x2)
        #print(x2.shape)256 33 33
        x3 = self.conv2(x3)
        #print(x3.shape)256 33 33
        x4 = self.conv2(x4)
        #print(x4.shape)256 33 33
        
        x1 = self.conv2(x1)
        #print(x1.shape)256 33 33
        x2 = self.conv2(x2)
        #print(x2.shape)256 33 33
        x3 = self.conv2(x3)
        #print(x3.shape)256 33 33
        x4 = self.conv2(x4)
        #print(x4.shape)256 33 33

        x5 = self.global_avg_pool(x)
        #print(x5.shape)256 1 1
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        #print(x.shape)1280 33 33

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)256 33 33

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_wasp(backbone, output_stride, BatchNorm):
    return Waterfall(backbone, output_stride, BatchNorm)
