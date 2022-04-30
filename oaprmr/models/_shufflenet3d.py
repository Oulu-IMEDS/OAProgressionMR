"""
Partially based on https://github.com/okankop/Efficient-3DCNNs/blob/master/models/shufflenet.py
Comparison against other archs: https://arxiv.org/pdf/1904.02422.pdf
See also "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"
"""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ShuffleNet3D(nn.Module):
    def __init__(self, *, config, path_weights, groups=3, width_mult=1):
        super(ShuffleNet3D, self).__init__()

        self.config = config

        if self.config["fe"]["l1_stride"] == 1:
            depth_stride = 1
            plane_stride = 1
        elif self.config["fe"]["l1_stride"] == 2:
            depth_stride = 2
            plane_stride = 2
        else:
            raise ValueError("Unsupported `l1_stride`")

        self.groups = groups
        num_blocks = [4, 8, 4]

        if groups == 1:
            out_planes = [24, 144, 288, 567]
        elif groups == 2:
            out_planes = [24, 200, 400, 800]
        elif groups == 3:
            out_planes = [24, 240, 480, 960]
        elif groups == 4:
            out_planes = [24, 272, 544, 1088]
        elif groups == 8:
            out_planes = [24, 384, 768, 1536]
        else:
            raise ValueError(f"{groups} groups is not supported")

        out_planes = [int(i * width_mult) for i in out_planes]
        self.in_planes = out_planes[0]
        self.conv1 = conv_bn(ch_in=self.config["input_channels"],
                             ch_out=out_planes[0],
                             stride=(depth_stride, plane_stride, plane_stride))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(out_planes[0], out_planes[1], num_blocks[0], self.groups)
        self.layer2 = self._make_layer(out_planes[1], out_planes[2], num_blocks[1], self.groups)
        self.layer3 = self._make_layer(out_planes[2], out_planes[3], num_blocks[2], self.groups)

        self.classifier = nn.Sequential(
            nn.Dropout(self.config["agg"]["dropout"]),
            nn.Linear(out_planes[3], self.config["output_channels"]))

        if self.config["restore_weights"]:
            self.load_state_dict(torch.load(path_weights))

    def _make_layer(self, ch_in, ch_out, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            if i == 0:
                layers.append(Bottleneck(ch_in, ch_out, stride=stride, groups=groups))
            else:
                layers.append(Bottleneck(ch_out, ch_out, stride=stride, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        endpoints = OrderedDict()

        # Code assumes: B, CH, S, R, C
        x = rearrange(x, "b ch r c s -> b ch s r c")

        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        endpoints["main"] = out
        return endpoints


def conv_bn(ch_in, ch_out, stride):
    return nn.Sequential(
        nn.Conv3d(ch_in, ch_out, kernel_size=(3, 3, 3), stride=stride,
                  padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(ch_out),
        nn.ReLU(inplace=True))


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.groups = groups
        mid_planes = out_planes // 4
        if self.stride == 2:
            out_planes = out_planes - in_planes
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv3d(in_planes, mid_planes, kernel_size=(1, 1, 1),
                               groups=g, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=(3, 3, 3),
                               stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, out_planes, kernel_size=(1, 1, 1),
                               groups=groups, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 2:
            self.shortcut = nn.AvgPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = rearrange(out, "b (g u) r c s -> b (u g) r c s", g=self.groups).contiguous()
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.stride == 2:
            out = self.relu(torch.cat([out, self.shortcut(x)], 1))
        else:
            out = self.relu(out + x)
        return out
