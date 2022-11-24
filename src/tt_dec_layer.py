import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import functools
from .SOC.skew_symmetric_conv import SOC


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def full_tt(K1, K2, K3):
    """Converts a TensorTrain into a regular tensor or matrix (tf.Tensor)."""
    res = K1
    K2_reshaped = torch.reshape(K2, (K2.shape[0], -1))
    res = torch.matmul(res, K2_reshaped)
    res = torch.reshape(res, (-1, K3.shape[0]))
    res = torch.matmul(res, K3)
    res = torch.reshape(res, (K1.shape[0],) + K2.shape[1:-1] + (K3.shape[-1],))
    num_dims = len(K2.shape[1:-1])
    return torch.permute(res, list(range(1, num_dims + 1)) + [0, num_dims + 1])


def faster_memory(convolution_op, inputs, K1, K2, K3):
    return convolution_op(
        inputs, torch.permute(full_tt(K1, K2, K3), (3, 2, 0, 1)).to(device)
    )


def slower_without_memory(convolution_op, inputs, K1, K2, K3):
    inputs1 = nn.functional.conv2d(inputs, K1.t().unsqueeze(-1).unsqueeze(-1))
    inputs2 = convolution_op(inputs1, torch.permute(K2, (3, 0, 1, 2)))
    return nn.functional.conv2d(inputs2, K3.t().unsqueeze(-1).unsqueeze(-1))


class ConvDecomposed2D_t(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        decomposition_rank,
        stride=1,
        padding=0,
        bias=True,
        device=None,
        dtype=None,
        use_memory=False,
        decomposition_scale=-1,
        **kwargs,
    ):
        assert (
            decomposition_scale == -1
            and decomposition_rank != -1
            or decomposition_rank == -1
            and decomposition_scale != -1
        )
        super(ConvDecomposed2D_t, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.decomposition_scale = decomposition_scale
        self.decomposition_rank = decomposition_rank

        if decomposition_scale != -1:
            r1 = int(self.in_channels * decomposition_scale)
            r2 = int(self.out_channels * decomposition_scale)
        else:
            r1 = min(self.in_channels, self.decomposition_rank)
            r2 = min(self.out_channels, self.decomposition_rank)

        self.r1 = r1
        self.r2 = r2

        self.K2 = nn.Parameter(
            torch.empty(
                (r1,) + self.kernel_size + (r2,),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        )

        if decomposition_rank >= in_channels and decomposition_rank >= out_channels:
            self.K1, self.K3 = None, None
        else:
            self.K1 = nn.Parameter(
                torch.empty(
                    self.in_channels, r1, dtype=dtype, device=device, requires_grad=True
                )
            )
            self.K3 = nn.Parameter(
                torch.empty(
                    r2,
                    self.out_channels,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
            )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.out_channels, dtype=dtype, device=device, requires_grad=True
                )
            )
        else:
            self.bias = None

        for weight in (self.K1, self.K2, self.K3):
            if weight is not None:
                nn.init.orthogonal_(weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.K2)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self._convolution_op = functools.partial(
            F.conv2d, stride=self.stride, padding=self.padding
        )
        self.use_memory = use_memory

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            + f"r1={self.r1}, r2={self.r2}, "
            + f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, "
            f"bias={self.bias is not None}"
        )

    def forward(self, inputs, use_memory=False):
        if self.K1 is None:
            outputs = self._convolution_op(inputs, torch.permute(self.K2, (3, 0, 1, 2)))
        elif self.use_memory or use_memory:
            outputs = faster_memory(
                self._convolution_op, inputs, self.K1, self.K2, self.K3
            )
        else:
            outputs = slower_without_memory(
                self._convolution_op, inputs, self.K1, self.K2, self.K3
            )
        if self.bias is not None:
            outputs += self.bias.view(1, -1, 1, 1)
        return outputs


def conv2tt(
    model,
    new_layer=ConvDecomposed2D_t,
    device=device,
    no_compress_first=False,
    **kwargs,
):
    for child_name, child in model.named_children():
        if "Conv2d" == child.__class__.__name__:
            # Hoping that the first conv is the first in order, which is not
            # guaranteed, but works for our WideResNet
            if child.groups != 1:
                raise ValueError("Groups are not supported in TT")
            if no_compress_first:
                no_compress_first = False
                continue
            add_kwargs = {}
            if new_layer not in (SOC,):
                add_kwargs["dilation"] = child.dilation
                add_kwargs["stride"] = child.stride
                add_kwargs["kernel_size"] = child.kernel_size
            else:
                add_kwargs["stride"] = min(2, child.stride[0])
                add_kwargs["kernel_size"] = child.kernel_size[0]

            setattr(
                model,
                child_name,
                new_layer(
                    child.in_channels,
                    child.out_channels,
                    padding=child.padding,
                    bias=(child.bias is not None),
                    **kwargs,
                    **add_kwargs,
                ).to(device),
            )
        else:
            conv2tt(child, new_layer, no_compress_first=no_compress_first, **kwargs)
