import torch
import torch.nn as nn

from .skew_symmetric_conv import SOC
from ..tt_dec_layer import ConvDecomposed2D_t
from ..SOTT import SOTT

conv_mapping = {
    "standard": nn.Conv2d,
    "skew": SOC,
    "tt": ConvDecomposed2D_t,
    "sott": SOTT,
}


class MinMax(nn.Module):
    def __init__(self):
        super(MinMax, self).__init__()

    def forward(self, z, axis=1):
        a, b = z.split(z.shape[axis] // 2, axis)
        c, d = torch.min(a, b), torch.max(a, b)
        return torch.cat([c, d], dim=axis)


def new_kwargs(kwargs, i):
    conv_kwargs = kwargs.copy()
    if "decomposition_rank" in kwargs.keys() and isinstance(
        kwargs["decomposition_rank"], list
    ):
        conv_kwargs["decomposition_rank"] = kwargs["decomposition_rank"][i]
    return conv_kwargs


class LipBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        conv_module,
        stride=1,
        kernel_size=3,
        block_n=-1,
        **kwargs
    ):
        super(LipBlock, self).__init__()
        self.activation = MinMax()
        conv_kwargs = new_kwargs(kwargs, block_n)
        self.conv = conv_module(
            in_planes,
            planes * stride,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            **conv_kwargs
        )
        if isinstance(self.conv, nn.Conv2d):
            nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.activation(self.conv(x))
        return x


class LipNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        conv_module,
        in_channels=32,
        num_classes=10,
        input_spatial_shape=32,
        **kwargs
    ):
        super(LipNet, self).__init__()
        self.activation = MinMax()
        self.in_planes = 3

        self.layer1 = self._make_layer(
            block,
            in_channels,
            num_blocks[0],
            conv_module,
            block_n=0,
            stride=2,
            kernel_size=3,
            **kwargs
        )
        self.layer2 = self._make_layer(
            block,
            self.in_planes,
            num_blocks[1],
            conv_module,
            block_n=1,
            stride=2,
            kernel_size=3,
            **kwargs
        )
        self.layer3 = self._make_layer(
            block,
            self.in_planes,
            num_blocks[2],
            conv_module,
            block_n=2,
            stride=2,
            kernel_size=3,
            **kwargs
        )
        self.layer4 = self._make_layer(
            block,
            self.in_planes,
            num_blocks[3],
            conv_module,
            block_n=3,
            stride=2,
            kernel_size=3,
            **kwargs
        )
        self.layer5 = self._make_layer(
            block,
            self.in_planes,
            num_blocks[4],
            conv_module,
            block_n=4,
            stride=2,
            kernel_size=1,
            **kwargs
        )

        flat_size = input_spatial_shape // 32
        flat_features = flat_size * flat_size * self.in_planes
        conv_kwargs = new_kwargs(kwargs, 5)
        self.final_conv = conv_module(
            flat_features, num_classes, kernel_size=1, stride=1, **conv_kwargs
        )

    def _make_layer(
        self, block, planes, num_blocks, conv_module, stride, kernel_size, **kwargs
    ):
        strides = [1] * (num_blocks - 1) + [stride]
        kernel_sizes = [3] * (num_blocks - 1) + [kernel_size]
        layers = []
        for stride, kernel_size in zip(strides, kernel_sizes):
            layers.append(
                block(
                    self.in_planes, planes, conv_module, stride, kernel_size, **kwargs
                )
            )
            self.in_planes = planes * stride
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        return x


def LipNet_n(
    conv_module_name,
    in_channels=32,
    num_blocks=4,
    num_classes=10,
    input_spatial_shape=32,
    **kwargs
):
    conv_module = conv_mapping[conv_module_name]
    num_blocks_list = [num_blocks] * 5
    return LipNet(
        LipBlock,
        num_blocks_list,
        conv_module,
        in_channels=in_channels,
        num_classes=num_classes,
        input_spatial_shape=input_spatial_shape,
        **kwargs
    )
