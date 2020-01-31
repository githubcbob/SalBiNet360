import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import B2_ResNet

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BasicConv2d_weight_initialize(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_weight_initialize, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MCM(nn.Module):
    # MCM-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(MCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),

            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))

        return x

class Aggregation(nn.Module):
    def __init__(self, channel):
        super(Aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d_weight_initialize(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d_weight_initialize(3*channel, 3*channel, 3, padding=1)
        self.conv5_weight_initialize = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5_weight_initialize(x)

        return x


class SalBiNet360(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32):
        super(SalBiNet360, self).__init__()
        self.resnet = B2_ResNet()

        self.mcm2_1 = MCM(512, channel)
        self.mcm3_1 = MCM(1024, channel)
        self.mcm4_1 = MCM(2048, channel)

        for p in self.parameters():
            p.requires_grad = False

        self.mcm4_2 = MCM(3584, 256)

        self.agg1 = Aggregation(channel)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(256, 256, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d_weight_initialize(64, 32, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d_weight_initialize(32, 1, 3, padding=1),
        )

        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.downsample_4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.downsample_2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)

        # global branch
        x2_1 = x2
        x3_1 = self.resnet.layer3_1(x2_1)
        x4_1 = self.resnet.layer4_1(x3_1)
        x2_1 = self.mcm2_1(x2_1)
        x3_1 = self.mcm3_1(x3_1)
        x4_1 = self.mcm4_1(x4_1)
        global_map = self.upsample_8(self.agg1(x4_1, x3_1, x2_1))

        # local branch
        x2_2_downsample = self.downsample_4(x2)

        x3_2 = self.resnet.layer3_2(x2)
        x3_2_downsample = self.downsample_2(x3_2)

        x4_2 = self.resnet.layer4_2(x3_2)
        x4_2 = torch.cat((x2_2_downsample, x3_2_downsample, x4_2), 1)
        x4_2 = self.mcm4_2(x4_2)

        local_map = self.decoder(x4_2)

        return global_map, local_map


