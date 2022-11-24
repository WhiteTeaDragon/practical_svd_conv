import argparse
import os
import wandb
import numpy as np
import torch
import pathlib
from torch import nn
from torchvision import transforms, datasets
from tqdm import tqdm
from time import time

from .tt_dec_layer import ConvDecomposed2D_t, conv2tt
from .SOTT import orthogonal_loss
from .utils import (
    constrain_conv,
    init_mapping,
    new_init_for_conv,
    clip_batch_norm,
    collect_sing_vals,
    operation_mapping,
    get_conv_output_shapes,
)
from .WideResNet import wide_resnet16_10
from .vgg import vgg19

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def count_acc(loader, criterion=nn.CrossEntropyLoss()):
    correct = 0
    total = 0
    total_loss = 0
    # since we're not training, we don't need to calculate the gradients for
    # our outputs
    model.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total, total_loss / len(loader)


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


conv_mapping = {"standard": nn.Conv2d, "tt": ConvDecomposed2D_t}

classes_mapping = {
    "cifar10": [10, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), datasets.CIFAR10],
    "cifar100": [
        100,
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
        datasets.CIFAR100,
    ],
}

arch_mapping = {"wrn16-10": wide_resnet16_10, "vgg19": vgg19}


def save_chp(epoch, model, optimizer, loss, args, best=False):
    checkpoints_dir = pathlib.Path(args.checkpoints_path)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    if best:
        epoch = "best"
    filename = (
        f"{args.architecture}_{args.new_layer}_rank_{args.dec_rank}_"
        f"clipping_{args.clipping}_clip_to_{args.clip_to}_"
        f"lip_bn_{args.lip_bn}_bn_eps_{args.bn_eps}_"
        f"epochs_{args.epochs}_opt_{args.opt}_init_lr_{args.init_lr}_"
        f"batch_size_{args.batch_size}_epoch_{epoch}_"
        f"ol_{args.orthogonal_k}_ds_{args.dec_scale}_"
        f"_dataset_{args.dataset}_"
        f"no-compress-first_{args.no_compress_first}.pth"
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoints_dir / filename,
    )


if __name__ == "__main__":
    """### Parsing arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--init-lr", default=0.01, type=float)
    parser.add_argument("--dec-rank", default=-1, type=int)
    parser.add_argument("--new-layer", default="standard", choices=conv_mapping.keys())
    parser.add_argument(
        "--clipping", default=None, choices=[None] + list(operation_mapping.keys())
    )
    parser.add_argument(
        "--architecture", default="wrn16-10", choices=arch_mapping.keys()
    )
    parser.add_argument(
        "--conv-init", default=None, choices=[None, *list(init_mapping.keys())]
    )
    parser.add_argument("--clip_freq", default=0, type=int)
    parser.add_argument("--clip_to", default=1, type=float)
    parser.add_argument("--clip_linear_to", default=-1, type=float)
    parser.add_argument("--opt", default="SGD", choices=["Adam", "SGD"])
    parser.add_argument("--weight-dec", default=1e-4, type=float)
    parser.add_argument("--bn-eps", default=1, type=float)
    parser.add_argument("--nesterov", dest="nesterov", action="store_true")
    parser.add_argument("--no-nesterov", dest="nesterov", action="store_false")
    parser.add_argument("--save_chp_every", default=200, type=int)
    parser.add_argument("--checkpoints-path", default="/home/checkpoints", type=str)
    parser.add_argument("--dataset-root", default="./data", type=str)
    parser.add_argument("--freq-bn", default=0, type=int)
    parser.add_argument("--multiply-by", default=1, type=float)
    parser.add_argument("--lip-bn", dest="lip_bn", action="store_true")
    parser.add_argument("--clip-linear", dest="clip_linear", action="store_true")
    parser.add_argument(
        "--collect-sing-vals", dest="collect_sing_vals", action="store_true"
    )
    parser.add_argument(
        "--no-compress-first", dest="no_compress_first", action="store_true"
    )
    parser.add_argument(
        "--gouk-transforms", dest="gouk_transforms", action="store_true"
    )
    parser.add_argument("--orthogonal-k", default=-1, type=float)
    parser.add_argument("--dec-scale", default=-1, type=float)
    parser.add_argument("--seed", default=3407, type=int)

    parser.set_defaults(
        nesterov=False,
        lip_bn=False,
        collect_sing_vals=False,
        clip_linear=False,
        no_compress_first=False,
        gouk_transforms=False,
    )

    args = parser.parse_args()
    assert (
        args.clipping is None
        and args.clip_freq == 0
        or args.clipping is not None
        and args.clip_freq > 0
    )
    assert (
        args.new_layer == "tt"
        and (args.dec_rank != -1 or args.dec_scale != -1)
        or args.new_layer != "tt"
        and args.dec_rank == -1
        and args.dec_scale == -1
    )
    assert args.lip_bn and args.freq_bn > 0 or not args.lip_bn and args.freq_bn == 0
    assert args.clipping != "divide-each-by-largest" or args.new_layer == "tt"
    assert args.orthogonal_k < 0 or "tt" in args.new_layer
    assert args.clip_linear_to < 0 or args.clip_linear

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """### Loading Data"""

    wandb.init(project="singular_values_resnet34")

    batch_size = args.batch_size
    num_classes, cifar_mean, cifar_std, dataset = classes_mapping[args.dataset]

    if args.architecture == "vgg19" or args.gouk_transforms:
        cifar_mean, cifar_std = 0.5, 0.5

    if args.gouk_transforms:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Pad(4, padding_mode="symmetric"),
                transforms.RandomCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(cifar_mean, cifar_std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(cifar_mean, cifar_std),
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    # load the data.
    trainset = dataset(
        root=args.dataset_root, train=True, download=True, transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=2
    )

    testset = dataset(
        root=args.dataset_root, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=2
    )

    epochs = args.epochs

    """#### Model without clipping"""

    kwargs = {}
    if "wrn" in args.architecture:
        kwargs["num_classes"] = num_classes
        model = arch_mapping[args.architecture](**kwargs).to(device)
    elif "vgg" in args.architecture:
        model = arch_mapping[args.architecture](num_classes=num_classes, **kwargs).to(
            device
        )
    else:
        model = arch_mapping[args.architecture](
            num_classes=num_classes, pretrained=False
        ).to(device)
    model_kwargs = {}
    if args.new_layer == "tt":
        if args.dec_scale != -1:
            model_kwargs["decomposition_rank"] = -1
            model_kwargs["decomposition_scale"] = args.dec_scale
        else:
            model_kwargs["decomposition_rank"] = args.dec_rank

    if args.new_layer != "standard":
        conv2tt(
            model,
            conv_mapping[args.new_layer],
            no_compress_first=args.no_compress_first,
            **model_kwargs,
        )
    if args.conv_init is not None:
        new_init_for_conv(model, args.conv_init)
    print(model)
    print(sum(p.numel() for p in model.parameters()))

    parameters = model.parameters()

    if args.opt == "Adam":
        if "vgg" in args.architecture:
            optimizer = torch.optim.Adam(
                parameters, lr=args.init_lr, amsgrad=True, eps=1e-7
            )
        else:
            optimizer = torch.optim.Adam(parameters, lr=args.init_lr)
    else:
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.init_lr,
            weight_decay=args.weight_dec,
            momentum=0.9,
            nesterov=args.nesterov,
        )
    lr_steps = epochs * len(train_loader)
    if "wrn" in args.architecture:
        assert args.epochs in [1, 200, 400]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                0.3 * args.epochs * len(train_loader),
                0.6 * args.epochs * len(train_loader),
                0.8 * args.epochs * len(train_loader),
            ],
            gamma=0.2,
        )
    elif "vgg" in args.architecture:
        assert args.epochs in [1, 140]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                (100 / 140) * args.epochs * len(train_loader),
                (120 / 140) * args.epochs * len(train_loader),
            ],
            gamma=0.1,
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[lr_steps // 2, (3 * lr_steps) // 4], gamma=0.1
        )
    criterion = nn.CrossEntropyLoss().to(device)
    count = 0
    if "lip" in args.architecture:
        conv_clip_assign_value = args.clip_to**0.5
        linear_clip_assign_value = 1
    else:
        conv_clip_assign_value = args.clip_to
        linear_clip_assign_value = None
        if args.clip_linear:
            linear_clip_assign_value = args.clip_to
            if args.clip_linear_to > 0:
                linear_clip_assign_value = args.clip_linear_to
    orthogonal = False

    conv_outputs = get_conv_output_shapes(model)

    best_val_acc = 0
    for epoch in range(epochs):
        clipping_during_epoch = 0
        running_loss = 0.0
        running_orthogonal_loss = 0.0
        opening = time()
        model.train()
        for i, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            count += 1
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(X)
            loss = criterion(output, y)
            if args.new_layer == "tt" and args.orthogonal_k > 0:
                orthogonal = True
                loss_, shapes_ = orthogonal_loss(model)
                running_orthogonal_loss += (loss_ / shapes_).item()
                loss += (args.orthogonal_k / shapes_) * loss_
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if args.lip_bn and count % args.freq_bn == 0:
                with torch.no_grad():
                    clip_batch_norm(model, args.bn_eps)
            if args.clip_freq != 0 and count % args.clip_freq == 0:
                start_time = time()
                with torch.no_grad():
                    constrain_conv(
                        model,
                        mode=args.clipping,
                        conv_clip_assign_value=conv_clip_assign_value,
                        linear_clip_assign_value=linear_clip_assign_value,
                        orthogonal=orthogonal,
                        outputs=conv_outputs,
                    )
                clipping_during_epoch += time() - start_time

        training_end = time()

        with torch.no_grad():
            if args.collect_sing_vals:
                collect_sing_vals(model)

        lr = scheduler.get_last_lr()[0]
        val_acc, val_loss = count_acc(test_loader)
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            save_chp(epoch, model, optimizer, loss, args, best=True)
        wandb.log(
            {
                "loss": running_loss / len(train_loader),
                "orthogonal_loss": running_orthogonal_loss / len(train_loader),
                "acc": count_acc(train_loader)[0],
                "val_acc": val_acc,
                "val_loss": val_loss,
                "epoch": epoch,
                "lr": lr,
                "time": time() - opening,
                "val_time": time() - training_end,
                "train_time": training_end - opening,
                "clipping_time": clipping_during_epoch,
            }
        )
        print("%d loss: %.3f" % (epoch + 1, running_loss / len(train_loader)))
        print(f"train acc: {count_acc(train_loader)}")
        if epoch != 0 and epoch % args.save_chp_every == 0:
            save_chp(epoch, model, optimizer, loss, args)

    print(f"test acc: {count_acc(test_loader)}")

    save_chp(epoch, model, optimizer, loss, args)
