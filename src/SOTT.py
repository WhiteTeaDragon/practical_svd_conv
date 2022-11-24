from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import einops


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def l2_normalize(tensor, eps=1e-12):
    norm = float(torch.sqrt(torch.sum(tensor.float() * tensor.float())))
    norm = max(norm, eps)
    ans = tensor / norm
    return ans


def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 0, 1)
    conv_filter_T = torch.flip(conv_filter_T, [2, 3])
    return conv_filter_T


def fantastic_four(conv_filter, num_iters=50):
    out_ch, in_ch, h, w = conv_filter.shape

    u1 = torch.randn((1, in_ch, 1, w), device=device, requires_grad=False)
    u1.data = l2_normalize(u1.data)

    u2 = torch.randn((1, in_ch, h, 1), device=device, requires_grad=False)
    u2.data = l2_normalize(u2.data)

    u3 = torch.randn((1, in_ch, h, w), device=device, requires_grad=False)
    u3.data = l2_normalize(u3.data)

    u4 = torch.randn((out_ch, 1, h, w), device=device, requires_grad=False)
    u4.data = l2_normalize(u4.data)

    v1 = torch.randn((out_ch, 1, h, 1), device=device, requires_grad=False)
    v1.data = l2_normalize(v1.data)

    v2 = torch.randn((out_ch, 1, 1, w), device=device, requires_grad=False)
    v2.data = l2_normalize(v2.data)

    v3 = torch.randn((out_ch, 1, 1, 1), device=device, requires_grad=False)
    v3.data = l2_normalize(v3.data)

    v4 = torch.randn((1, in_ch, 1, 1), device=device, requires_grad=False)
    v4.data = l2_normalize(v4.data)

    for i in range(num_iters):
        v1.data = l2_normalize(
            (conv_filter.data * u1.data).sum((1, 3), keepdim=True).data
        )
        u1.data = l2_normalize(
            (conv_filter.data * v1.data).sum((0, 2), keepdim=True).data
        )

        v2.data = l2_normalize(
            (conv_filter.data * u2.data).sum((1, 2), keepdim=True).data
        )
        u2.data = l2_normalize(
            (conv_filter.data * v2.data).sum((0, 3), keepdim=True).data
        )

        v3.data = l2_normalize(
            (conv_filter.data * u3.data).sum((1, 2, 3), keepdim=True).data
        )
        u3.data = l2_normalize((conv_filter.data * v3.data).sum(0, keepdim=True).data)

        v4.data = l2_normalize(
            (conv_filter.data * u4.data).sum((0, 2, 3), keepdim=True).data
        )
        u4.data = l2_normalize((conv_filter.data * v4.data).sum(1, keepdim=True).data)

    return u1, v1, u2, v2, u3, v3, u4, v4


class SOTT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        decomposition_rank,
        stride=1,
        padding=0,
        bias=True,
        device=device,
        dtype=None,
        train_terms=5,
        eval_terms=12,
        init_iters=50,
        update_iters=1,
        update_freq=200,
        correction=0.7,
        decomposition_scale=-1,
        **kwargs,
    ):
        super(SOTT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, tuple):
            stride = stride[0]
        assert stride in (1, 2)

        self.stride = stride
        self.padding = padding

        self.init_iters = init_iters
        self.update_iters = update_iters
        self.update_freq = update_freq
        self.total_iters = 0
        self.train_terms = train_terms
        self.eval_terms = eval_terms

        # ranks
        assert (
            decomposition_scale == -1
            and decomposition_rank != -1
            or decomposition_rank == -1
            and decomposition_scale != -1
        )

        self.decomposition_scale = decomposition_scale
        self.decomposition_rank = decomposition_rank

        if decomposition_scale != -1:
            r1 = int(self.in_channels * decomposition_scale)
            r2 = int(self.out_channels * decomposition_scale)
        else:
            r1 = min(self.in_channels, self.decomposition_rank)
            r2 = min(self.out_channels, self.decomposition_rank)

        # if strides > 1, SOC reshapes input to have more channels, and compressed w & h
        self.max_r = max(r1 * stride * stride, r2)
        self.r1_s, self.r2 = r1 * stride * stride, r2

        self.K2 = nn.Parameter(
            torch.empty(
                (self.max_r, self.max_r) + self.kernel_size,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        )

        # init as it is in SOC
        stdv = 1.0 / np.sqrt(self.max_r)
        nn.init.normal_(self.K2, std=stdv)

        random_conv_filter_T = transpose_filter(self.K2)
        conv_filter = 0.5 * (self.K2 - random_conv_filter_T)

        with torch.no_grad():
            u1, v1, u2, v2, u3, v3, u4, v4 = fantastic_four(
                conv_filter, num_iters=self.init_iters
            )
            self.u1 = nn.Parameter(u1, requires_grad=False)
            self.v1 = nn.Parameter(v1, requires_grad=False)
            self.u2 = nn.Parameter(u2, requires_grad=False)
            self.v2 = nn.Parameter(v2, requires_grad=False)
            self.u3 = nn.Parameter(u3, requires_grad=False)
            self.v3 = nn.Parameter(v3, requires_grad=False)
            self.u4 = nn.Parameter(u4, requires_grad=False)
            self.v4 = nn.Parameter(v4, requires_grad=False)

        self.correction = nn.Parameter(
            torch.Tensor([correction]).to(device), requires_grad=False
        )

        if (
            decomposition_rank != -1
            and decomposition_rank >= in_channels
            and decomposition_rank >= out_channels
        ):
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

        for weight in (self.K1, self.K3):
            if weight is not None:
                nn.init.orthogonal_(weight)

        # init as it is in SOC
        if self.bias is not None:
            stdv = 1.0 / np.sqrt(self.out_channels)
            nn.init.uniform_(self.bias, -stdv, stdv)

    def update_sigma(self):
        if self.training:
            if self.total_iters % self.update_freq == 0:
                update_iters = self.init_iters
            else:
                update_iters = self.update_iters
            self.total_iters = self.total_iters + 1
        else:
            update_iters = 0

        random_conv_filter_T = transpose_filter(self.K2)
        conv_filter = 0.5 * (self.K2 - random_conv_filter_T)

        with torch.no_grad():
            for i in range(update_iters):
                self.v1.data = l2_normalize(
                    (conv_filter * self.u1).sum((1, 3), keepdim=True).data
                )
                self.u1.data = l2_normalize(
                    (conv_filter * self.v1).sum((0, 2), keepdim=True).data
                )
                self.v2.data = l2_normalize(
                    (conv_filter * self.u2).sum((1, 2), keepdim=True).data
                )
                self.u2.data = l2_normalize(
                    (conv_filter * self.v2).sum((0, 3), keepdim=True).data
                )
                self.v3.data = l2_normalize(
                    (conv_filter * self.u3).sum((1, 2, 3), keepdim=True).data
                )
                self.u3.data = l2_normalize(
                    (conv_filter * self.v3).sum(0, keepdim=True).data
                )
                self.v4.data = l2_normalize(
                    (conv_filter * self.u4).sum((0, 2, 3), keepdim=True).data
                )
                self.u4.data = l2_normalize(
                    (conv_filter * self.v4).sum(1, keepdim=True).data
                )

        func = torch.min
        sigma1 = torch.sum(conv_filter * self.u1 * self.v1)
        sigma2 = torch.sum(conv_filter * self.u2 * self.v2)
        sigma3 = torch.sum(conv_filter * self.u3 * self.v3)
        sigma4 = torch.sum(conv_filter * self.u4 * self.v4)
        sigma = func(func(func(sigma1, sigma2), sigma3), sigma4)
        return sigma

    def extra_repr(self):
        repr = (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, padding={self.padding}, "
            f"stride={self.stride}, "
            f"bias={self.bias is not None}, "
            f"train_terms={self.train_terms}, eval_terms={self.eval_terms}"
        )
        if self.decomposition_scale == -1:
            return f"rank={self.decomposition_rank}, " + repr
        else:
            return f"scale={self.decomposition_scale}, " + repr

    def forward(self, x):
        random_conv_filter_T = transpose_filter(self.K2)
        conv_filter_skew = 0.5 * (self.K2 - random_conv_filter_T)

        sigma = self.update_sigma()
        conv_filter_n = (self.correction * conv_filter_skew) / sigma

        if self.training:
            num_terms = self.train_terms
        else:
            num_terms = self.eval_terms

        if self.K1 is not None:
            x = nn.functional.conv2d(x, self.K1.t().unsqueeze(-1).unsqueeze(-1))

        if self.stride > 1:
            x = einops.rearrange(
                x,
                "b c (w k1) (h k2) -> b (c k1 k2) w h",
                k1=self.stride,
                k2=self.stride,
            )

        # in SOC, you work with square kernels,
        # so you have to pad input channels
        if self.r2 > self.r1_s:
            diff_channels = self.r2 - self.r1_s
            p4d = (0, 0, 0, 0, 0, diff_channels, 0, 0)
            x = F.pad(x, p4d)

        # the main idea of SOC
        curr_z = x
        output = x
        for i in range(1, num_terms + 1):
            curr_z = F.conv2d(
                curr_z,
                conv_filter_n,
                padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2),
            ) / float(i)
            output = output + curr_z

        # since the kernel is square,
        # in this case you have to drop some channels
        if self.r2 < self.r1_s:
            output = output[:, : self.r2, :, :]

        if self.K3 is not None:
            output = nn.functional.conv2d(
                output, self.K3.t().unsqueeze(-1).unsqueeze(-1)
            )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output


def orthogonal_loss(model):
    loss, shapes = 0, 0
    for child_name, child in model.named_children():
        if (
            "TT" in child.__class__.__name__
            or "ConvDecomposed" in child.__class__.__name__
        ) and child.K1 is not None:
            loss += nn.functional.mse_loss(
                child.K1.T @ child.K1,
                torch.eye(child.K1.shape[1]).to(device),
                reduction="sum",
            )
            loss += nn.functional.mse_loss(
                child.K3 @ child.K3.T,
                torch.eye(child.K3.shape[0]).to(device),
                reduction="sum",
            )
            shapes += 2 * child.K3.shape[0] ** 2
        else:
            loss_, shapes_ = orthogonal_loss(child)
            loss += loss_
            shapes += shapes_
    return loss, shapes
