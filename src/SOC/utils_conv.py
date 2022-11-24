import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 0, 1)
    conv_filter_T = torch.flip(conv_filter_T, [2, 3])
    return conv_filter_T


def l2_normalize(tensor, eps=1e-12):
    norm = float(torch.sqrt(torch.sum(tensor.float() * tensor.float())))
    norm = max(norm, eps)
    ans = tensor / norm
    return ans


def real_power_iteration(
    conv_filter, inp_shape=(32, 32), num_iters=50, return_uv=False
):
    H, W = inp_shape
    c_out = conv_filter.shape[0]
    c_in = conv_filter.shape[1]
    pad_size = conv_filter.shape[2] // 2
    with torch.no_grad():
        u = l2_normalize(
            torch.randn(1, c_out, H, W, dtype=conv_filter.dtype).to(device).data
        )
        v = l2_normalize(
            torch.randn(1, c_in, H, W, dtype=conv_filter.dtype).to(device).data
        )

        for i in range(num_iters):
            v.data = l2_normalize(
                F.conv_transpose2d(u.data, conv_filter.data, padding=pad_size)
            )
            u.data = l2_normalize(F.conv2d(v.data, conv_filter.data, padding=pad_size))
        sigma = torch.sum(u.data * F.conv2d(v.data, conv_filter.data, padding=pad_size))
    if return_uv:
        return sigma, u, v
    else:
        return sigma


def test_real_sn(model):
    model_l = [module for module in model.modules() if type(module) != nn.Sequential]
    sigma_list = []
    variants = [
        "<class 'skew_symmetric_conv.skew_conv'>",
        "<class 'lip_skew_symmetric_conv.skew_conv'>",
    ]
    for module in model_l:
        if str(type(module)) in variants:
            conv_filter = module.random_conv_filter
            conv_filter_T = transpose_filter(conv_filter)
            conv_filter_skew = 0.5 * (conv_filter - conv_filter_T)

            real_sigma = module.update_sigma()
            conv_filter_n = (module.correction * conv_filter_skew) / real_sigma

            real_sigma = real_power_iteration(conv_filter_n, num_iters=50)

            sigma_list.append(real_sigma.detach().cpu().item())
    sigma_array = np.array(sigma_list)
    return sigma_array
