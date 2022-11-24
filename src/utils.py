import io
from contextlib import redirect_stdout

import torch
import torchvision
from .tt_dec_layer import ConvDecomposed2D_t, full_tt
import wandb
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchsummary as ts
from einops import rearrange


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def get_new_K(K1, K2, K3):
    # (m, R) -> (m, min(m, R)) -- q: q.T @ q = I, (min(m, R), R) -- r
    q1, r1 = torch.linalg.qr(K1)
    q3, r3 = torch.linalg.qr(torch.transpose(K3, -1, -2))
    middle_k = full_tt(r1, K2, torch.transpose(r3, -1, -2))
    return q1, middle_k, torch.transpose(q3, -1, -2)


def assigning(D, assign):
    return torch.ones(D.shape).cpu() * assign


def clipping(D, clip_to):
    return torch.clamp(D, max=clip_to)


def divide_by_sv(
    w,
    max_k,
    in_shape,
    in_channels,
    stride=(1, 1),
    padding=(0, 0),
    iterations=2,
    mode="divide-by-largest",
):
    """
    in_shape: (c, h, w)
    """
    if len(w.shape) == 4:
        if "exact" in mode:
            before_svd = get_ready_for_svd(
                w.cpu().permute([2, 3, 0, 1]), in_shape, stride
            )
            norm = torch.linalg.svdvals(before_svd[-1])[:, :, 0].max()
        else:
            norm = power_iteration(
                in_channels, in_shape, iterations, padding, stride, w
            )
    else:
        if "exact" in mode:
            norm = torch.linalg.svdvals(w.cpu())[0]
        else:
            norm = power_iteration_linear(iterations, w)

    return w * (1.0 / max(1.0, norm.item() / max_k))


def power_iteration_linear(iterations, w):
    x = torch.normal(size=(int(w.shape[1]), 1), mean=0, std=1).to(device)
    for i in range(0, iterations):
        x_p = torch.matmul(w, x)
        x = torch.matmul(w.t(), x_p)
    norm = torch.sqrt(
        torch.sum(torch.pow(torch.matmul(w, x), 2.0)) / torch.sum(torch.pow(x, 2.0))
    )
    return norm


def power_iteration(in_channels, in_shape, iterations, padding, stride, w):
    x = torch.normal(
        size=(
            1,
            in_channels,
        )
        + in_shape,
        mean=0,
        std=1,
    ).to(device)
    for i in range(iterations):
        x_p = F.conv2d(x, w, stride=stride, padding=padding)
        x = F.conv_transpose2d(x_p, w, stride=stride, padding=padding)
    Wx = F.conv2d(x, w, stride=stride, padding=padding)
    norm = torch.sqrt(torch.sum(torch.pow(Wx, 2.0)) / torch.sum(torch.pow(x, 2.0)))
    return norm


operation_mapping = {
    "assign": assigning,
    "clip": clipping,
    "divide-by-largest": clipping,
    "divide-by-exact-largest": clipping,
    "divide-each-by-largest": clipping,
}


def singular_value_operation(
    number, transform_coeff, operation, cast_to_complex=True, wandb_name=None
):
    U, D, V = torch.linalg.svd(transform_coeff, full_matrices=False)
    D_clipped = operation(D, number)
    if wandb_name is not None:
        wandb.log({wandb_name: wandb.Histogram(D)})
    if cast_to_complex:
        D_clipped = D_clipped.type(torch.complex64)
    D_clipped = torch.diag_embed(D_clipped)
    clipped_coeff = torch.linalg.matmul(D_clipped, V)
    clipped_coeff = torch.linalg.matmul(U, clipped_coeff)
    return clipped_coeff


def get_ready_for_svd(kernel, pad_to, strides):
    assert len(kernel.shape) == 4  # K2 is given with shape (k, k, r1, r2)
    assert len(pad_to) == len(kernel.shape) - 2
    dim = 2
    if isinstance(strides, int):
        strides = [strides] * dim
    else:
        assert len(strides) == dim
    for i in range(dim):
        assert pad_to[i] % strides[i] == 0
        assert kernel.shape[i] <= pad_to[i]
    old_shape = kernel.shape
    kernel_tr = torch.permute(kernel, dims=[dim, dim + 1] + list(range(dim)))
    padding_tuple = []
    for i in range(dim):
        padding_tuple.append(0)
        padding_tuple.append(pad_to[-i - 1] - kernel_tr.shape[-i - 1])
    kernel_pad = torch.nn.functional.pad(kernel_tr, tuple(padding_tuple))
    r1, r2 = kernel_pad.shape[:2]
    small_shape = []
    for i in range(dim):
        small_shape.append(pad_to[i] // strides[i])
    reshape_for_fft = torch.zeros(
        (r1, r2, np.prod(np.array(strides))) + tuple(small_shape)
    )
    for i in range(strides[0]):
        for j in range(strides[1]):
            reshape_for_fft[:, :, i * strides[1] + j, :, :] = kernel_pad[
                :, :, i :: strides[0], j :: strides[1]
            ]
    fft_results = torch.fft.fft2(reshape_for_fft).reshape(r1, -1, *small_shape)
    # sing_vals shape is (r1, 4r2, k, k, k)
    transpose_for_svd = np.transpose(fft_results, axes=list(range(2, dim + 2)) + [0, 1])
    # now the shape is (k, k, k, r1, 4r2)
    return kernel_pad, old_shape, r1, r2, small_shape, strides, transpose_for_svd


def Clip_AssignOperatorNorm(
    kernel, pad_to, number, strides, operation=clipping, wandb_name=None
):
    if kernel.shape[0] > pad_to[0]:
        k, n = kernel.shape[0], pad_to[0]
        assert k == n + 2 or k == n + 1
        pad_kernel = torch.nn.functional.pad(
            kernel, (0, 0, 0, 0, 0, max(k, 2 * n) - k, 0, max(k, 2 * n) - k)
        )
        tmp = rearrange(
            pad_kernel,
            "(w1 k1) (w2 k2) cin cout -> (k1 k2) (w1 w2) cin cout",
            w1=2,
            w2=2,
        )
        sv = torch.sqrt((tmp.sum(1) ** 2).sum(0))
        coef = (sv > number) * sv + (sv <= number)
        coef = coef.repeat(kernel.shape[0], kernel.shape[0], 1, 1)
        return kernel / coef

    (
        kernel_pad,
        old_shape,
        r1,
        r2,
        small_shape,
        strides,
        transpose_for_svd,
    ) = get_ready_for_svd(kernel, pad_to, strides)
    dim = len(small_shape)
    clipped_sing_vals = singular_value_operation(
        number, transpose_for_svd, operation, wandb_name=wandb_name
    )
    clipped_sing_vals = torch.permute(
        clipped_sing_vals, dims=[dim, dim + 1] + list(range(dim))
    )
    # now the shape is (r1, 4r2, k, k, k)
    clipped_sing_vals = clipped_sing_vals.reshape(r1, r2, -1, *small_shape)
    # now the shape is (r1, r2, -1, k, k, k)
    kernel = torch.fft.ifft2(clipped_sing_vals).real
    for i in range(strides[0]):
        for j in range(strides[1]):
            kernel_pad[:, :, i :: strides[0], j :: strides[1]] = kernel[
                :, :, i * strides[1] + j, :, :
            ]
    # now the shape is (r1, r2, k, k, k)
    kernel = torch.permute(kernel_pad, dims=list(range(2, dim + 2)) + [0, 1])[
        : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
    ]
    return kernel


def get_input_shape(layer, output_shape):
    if isinstance(layer.stride, int):
        return np.array(output_shape) * layer.stride
    return tuple(np.array(output_shape) * np.array(layer.stride))


def constrain_conv(
    model,
    conv_clip_assign_value=1,
    linear_clip_assign_value=None,
    mode="assign",
    iterations=2,
    orthogonal=False,
    outputs=[],
):
    count = 0
    index = -1
    operation = operation_mapping[mode]
    for layer in model.modules():
        if isinstance(layer, ConvDecomposed2D_t):
            index += 1
            if mode == "divide-each-by-largest":
                if layer.K1 is not None:
                    K1 = divide_by_sv(
                        layer.K1.t().unsqueeze(-1).unsqueeze(-1),
                        conv_clip_assign_value,
                        get_input_shape(layer, outputs[index]),
                        layer.in_channels,
                        stride=layer.stride,
                        padding=layer.padding,
                        iterations=iterations,
                        mode=mode,
                    )
                    layer.K1.data = K1.squeeze(-1).squeeze(-1).t().data
                K2 = divide_by_sv(
                    layer.K2.permute([3, 0, 1, 2]),
                    conv_clip_assign_value,
                    get_input_shape(layer, outputs[index]),
                    min(layer.in_channels, layer.decomposition_rank),
                    stride=layer.stride,
                    padding=layer.padding,
                    iterations=iterations,
                    mode=mode,
                )
                layer.K2.data = K2.permute([1, 2, 3, 0])
                if layer.K3 is not None:
                    K3 = divide_by_sv(
                        layer.K3.t().unsqueeze(-1).unsqueeze(-1),
                        conv_clip_assign_value,
                        get_input_shape(layer, outputs[index]),
                        layer.decomposition_rank,
                        stride=layer.stride,
                        padding=layer.padding,
                        iterations=iterations,
                        mode=mode,
                    )
                    layer.K3.data = K3.squeeze(-1).squeeze(-1).t().data
                continue
            if layer.K1 is not None and not orthogonal:
                K1, K2, K3 = get_new_K(layer.K1, layer.K2, layer.K3)
                layer.K1.data = K1.data
                layer.K3.data = K3.data
            else:
                K2 = torch.permute(layer.K2, dims=[1, 2, 0, 3])
            if mode.startswith("divide-by"):
                K2 = K2.permute([3, 2, 0, 1])
                K2 = divide_by_sv(
                    K2,
                    conv_clip_assign_value,
                    get_input_shape(layer, outputs[index]),
                    min(layer.in_channels, layer.decomposition_rank),
                    stride=layer.stride,
                    padding=layer.padding,
                    iterations=iterations,
                    mode=mode,
                )
                K2 = K2.permute([2, 3, 1, 0])
            else:
                K2 = K2.cpu()
                K2 = Clip_AssignOperatorNorm(
                    K2,
                    get_input_shape(layer, outputs[index]),
                    conv_clip_assign_value,
                    layer.stride,
                    operation=operation,
                    wandb_name=None,
                )
            K2 = torch.permute(K2, [2, 0, 1, 3])
            K2 = K2.to(device)
            layer.K2.data = K2.data
            count += 1
        elif isinstance(layer, nn.Conv2d):
            index += 1
            if mode.startswith("divide-by"):
                new_weight = layer.weight
                new_weight = divide_by_sv(
                    new_weight,
                    conv_clip_assign_value,
                    get_input_shape(layer, outputs[index]),
                    layer.in_channels,
                    stride=layer.stride,
                    padding=layer.padding,
                    iterations=iterations,
                    mode=mode,
                )
            else:
                new_weight = layer.weight.cpu()
                new_weight = torch.permute(new_weight, dims=[2, 3, 0, 1])
                new_weight = Clip_AssignOperatorNorm(
                    new_weight,
                    get_input_shape(layer, outputs[index]),
                    conv_clip_assign_value,
                    layer.stride,
                    operation=operation,
                    wandb_name=None,
                )
                new_weight = torch.permute(new_weight, dims=[2, 3, 0, 1]).to(device)
            layer.weight.data = new_weight.data
        elif isinstance(layer, nn.Linear) and linear_clip_assign_value is not None:
            if mode.startswith("divide"):
                new_weight = layer.weight
                new_weight = divide_by_sv(
                    new_weight,
                    linear_clip_assign_value,
                    None,
                    None,
                    iterations=iterations,
                    mode=mode,
                )
            else:
                new_weight = layer.weight.cpu()
                new_weight = singular_value_operation(
                    linear_clip_assign_value, new_weight, operation, False
                )
            layer.weight.data = new_weight.to(device).data


def collect_sing_vals(model, input_shape=(32, 32), iterations=2):
    count = 0
    bn_count = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            max_sing_power_iter = power_iteration(
                layer.in_channels,
                input_shape,
                iterations,
                layer.padding,
                layer.stride,
                layer.weight,
            )
            wandb.log({f"power_it_sv_{count}_layer": max_sing_power_iter})
            before_svd = get_ready_for_svd(
                layer.weight.cpu().permute([2, 3, 0, 1]), input_shape, layer.stride
            )
            max_sing_true = torch.linalg.svdvals(before_svd[-1])[:, :, 0].max()
            wandb.log({f"true_sv_{count}_layer": max_sing_true})
            wandb.log(
                {f"true_minus_pi_{count}_layer": max_sing_true - max_sing_power_iter}
            )
            count += 1
        elif isinstance(layer, nn.Linear):
            max_sing_power_iter = power_iteration_linear(iterations, layer.weight)
            wandb.log({f"power_it_sv_linear_layer": max_sing_power_iter})
            max_sing_true = torch.linalg.svdvals(layer.weight.cpu())[0]
            wandb.log({f"true_sv_linear_layer": max_sing_true})
            wandb.log(
                {f"true_minus_pi_linear_layer": max_sing_true - max_sing_power_iter}
            )
        elif isinstance(layer, nn.BatchNorm2d):
            norm = torch.max(
                torch.abs(layer.weight / torch.sqrt(layer.running_var + 1e-6))
            )
            wandb.log({f"bn_{bn_count}_layer": norm})
            bn_count += 1


def clip_batch_norm(model, max_k):
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            norm = torch.max(
                torch.abs(layer.weight / torch.sqrt(layer.running_var + 1e-6))
            )
            layer.weight.copy_(layer.weight / torch.clamp(norm / max_k, min=1.0))


def get_testloader(
    batch_size, dataset_root="./data", dataset="cifar10", gouk_transforms=False
):
    if dataset == "cifar10":
        testset = create_cifar10_testset(dataset_root, gouk_transforms=gouk_transforms)
    else:
        testset = create_cifar100_testset(dataset_root, gouk_transforms=gouk_transforms)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=0
    )
    return testloader


def create_cifar10_testset(dataset_root, gouk_transforms=False):
    cifar_mean, cifar_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if gouk_transforms:
        cifar_mean, cifar_std = 0.5, 0.5
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(cifar_mean, cifar_std),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root=dataset_root, train=False, download=True, transform=test_transform
    )
    return testset


def create_cifar100_testset(dataset_root, gouk_transforms=False):
    cifar_mean, cifar_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    if gouk_transforms:
        cifar_mean, cifar_std = 0.5, 0.5
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(cifar_mean, cifar_std),
        ]
    )
    testset = torchvision.datasets.CIFAR100(
        root=dataset_root, train=False, download=True, transform=test_transform
    )
    return testset


init_mapping = {
    "orthogonal": nn.init.orthogonal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
}


def new_init_for_conv(model, new_init="orthogonal", **kwargs):
    for child_name, child in model.named_children():
        if "Conv" in child.__class__.__name__:
            if "Conv2d" == child.__class__.__name__:
                init_mapping[new_init](child.weight)
            elif "Decomposed2D" in child.__class__.__name__:
                for weight in (child.K1, child.K2, child.K3):
                    if weight is not None:
                        init_mapping[new_init](weight)
            else:
                print(
                    f"Unrecognized layer {child.__class__.__name__}, "
                    "skipped re-initialization"
                )
        else:
            new_init_for_conv(child, new_init, **kwargs)


def get_conv_output_shapes(model, shape=(3, 32, 32)):
    with io.StringIO() as buf, redirect_stdout(buf):
        ts.summary(model, shape)
        output = buf.getvalue()
    conv_lines = [line for line in output.split("\n") if "Conv" in line]
    conv_outputs = [
        list(map(int, line.split("[")[1].split("]")[0].split(",")))[2:]
        for line in conv_lines
    ]
    return conv_outputs
