"""Partially based on https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet2p1d.py
See also https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf"""

from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResNet2P1D(nn.Module):
    def __init__(self, *, config, path_weights, block, layers, block_inplanes,
                 conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, shortcut_type='B',
                 widen_factor=1.0):
        super(ResNet2P1D, self).__init__()

        self.config = config

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv1_s = nn.Conv3d(self.config["input_channels"], mid_planes,
                                 kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                 padding=(0, 3, 3), bias=False)
        self.bn1_s = nn.BatchNorm3d(mid_planes)
        self.conv1_t = nn.Conv3d(mid_planes, self.in_planes,
                                 kernel_size=(conv1_t_size, 1, 1), stride=(conv1_t_stride, 1, 1),
                                 padding=(conv1_t_size // 2, 0, 0), bias=False)
        self.bn1_t = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1],
                                       shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2],
                                       shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3],
                                       shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion,
                            self.config["output_channels"])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.config["restore_weights"]:
            self.load_state_dict(torch.load(path_weights))

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0),
                                planes - out.size(1),
                                out.size(2),
                                out.size(3),
                                out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes, planes=planes,
                  stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        endpoints = OrderedDict()

        # Code assumes: B, CH, S, R, C
        x = rearrange(x, "b ch r c s -> b ch s r c")

        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        endpoints["main"] = x
        return endpoints


def conv1x3x3(in_planes, mid_planes, stride=1):
    return nn.Conv3d(in_planes, mid_planes, kernel_size=(1, 3, 3),
                     stride=(1, stride, stride), padding=(0, 1, 1), bias=False)


def conv3x1x1(mid_planes, planes, stride=1):
    return nn.Conv3d(mid_planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                     padding=(1, 0, 0), bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1
        self.conv1_s = conv1x3x3(in_planes, mid_planes1, stride)
        self.bn1_s = nn.BatchNorm3d(mid_planes1)
        self.conv1_t = conv3x1x1(mid_planes1, planes, stride)
        self.bn1_t = nn.BatchNorm3d(planes)

        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(planes, mid_planes2)
        self.bn2_s = nn.BatchNorm3d(mid_planes2)
        self.conv2_t = conv3x1x1(mid_planes2, planes)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)

        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2p1d_parameters = planes * 3 * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv2_s = conv1x3x3(planes, mid_planes, stride)
        self.bn2_s = nn.BatchNorm3d(mid_planes)
        self.conv2_t = conv3x1x1(mid_planes, planes, stride)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def ResNet2P1D10(**kwargs):
    model = ResNet2P1D(block=BasicBlock, layers=[1, 1, 1, 1],
                       block_inplanes=[64, 128, 256, 512], **kwargs)
    return model


def ResNet2P1D18(**kwargs):
    model = ResNet2P1D(block=BasicBlock, layers=[2, 2, 2, 2],
                       block_inplanes=[64, 128, 256, 512], **kwargs)
    return model


def ResNet2P1D34(**kwargs):
    model = ResNet2P1D(block=BasicBlock, layers=[3, 4, 6, 3],
                       block_inplanes=[64, 128, 256, 512], **kwargs)
    return model


def ResNet2P1D50(**kwargs):
    model = ResNet2P1D(block=BasicBlock, layers=[3, 4, 6, 3],
                       block_inplanes=[64, 128, 256, 512], **kwargs)
    return model


def ResNet2P1D101(**kwargs):
    model = ResNet2P1D(block=BasicBlock, layers=[3, 4, 23, 3],
                       block_inplanes=[64, 128, 256, 512], **kwargs)
    return model


def ResNet2P1D152(**kwargs):
    model = ResNet2P1D(block=BasicBlock, layers=[3, 8, 36, 3],
                       block_inplanes=[64, 128, 256, 512], **kwargs)
    return model


def ResNet2P1D200(**kwargs):
    model = ResNet2P1D(block=BasicBlock, layers=[3, 24, 36, 3],
                       block_inplanes=[64, 128, 256, 512], **kwargs)
    return model
