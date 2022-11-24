# copied from https://github.com/yoshitomo-matsubara/torchdistill/blob/f7e757f63982c76bc3e9db4354edf1e008963f8a/torchdistill/models/classification/wide_resnet.py
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride, conv_shortcut):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.shortcut = nn.Sequential()
        if conv_shortcut:
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, planes, kernel_size=1, stride=stride, bias=False
                    ),
                )
        else:
            shortcut_array = []
            if stride != 1:
                shortcut_array.append(nn.AvgPool2d(stride))
            if in_planes != planes:
                diff = planes - in_planes
                shortcut_array.append(
                    LambdaLayer(
                        lambda x: nn.functional.pad(
                            x, (0, 0, 0, 0, diff - diff // 2, diff // 2)
                        )
                    )
                )
            self.shortcut = nn.Sequential(*shortcut_array)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(
        self,
        depth,
        k,
        dropout_p,
        block,
        num_classes,
        norm_layer=None,
        conv_shortcut=True,
    ):
        super().__init__()
        n = (depth - 4) / 6
        stage_sizes = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = 16
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(
            3, stage_sizes[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_wide_layer(
            block, stage_sizes[1], n, dropout_p, 1, conv_shortcut
        )
        self.layer2 = self._make_wide_layer(
            block, stage_sizes[2], n, dropout_p, 2, conv_shortcut
        )
        self.layer3 = self._make_wide_layer(
            block, stage_sizes[3], n, dropout_p, 2, conv_shortcut
        )
        self.bn1 = norm_layer(stage_sizes[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_sizes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_wide_layer(
        self, block, planes, num_blocks, dropout_rate, stride, conv_shortcut
    ):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, dropout_rate, stride, conv_shortcut)
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def wide_resnet(
    depth: int, k: int, dropout_p: float, num_classes: int, **kwargs: Any
) -> WideResNet:
    assert (depth - 4) % 6 == 0, "depth of Wide ResNet (WRN) should be 6n + 4"
    model = WideResNet(depth, k, dropout_p, WideBasicBlock, num_classes, **kwargs)
    return model


def wide_resnet16_10(num_classes=100, **kwargs: Any) -> WideResNet:
    return wide_resnet(16, 10, 0, num_classes, conv_shortcut=False, **kwargs)
