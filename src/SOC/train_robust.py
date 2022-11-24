import argparse
import copy
import logging
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from .robust_net import LipNet_n
from .utils import cifar10_std, get_loaders, evaluate_certificates
from .utils_conv import test_real_sn
from ..SOTT import orthogonal_loss
from ..utils import constrain_conv

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

logger = logging.getLogger(__name__)


def max_sv(matrix):
    U, D, V = torch.linalg.svd(matrix, full_matrices=False)
    return D.max().item()


def max_conv_lip(model, cur_max=0):
    for child_name, child in model.named_children():
        if "SOTT" in child.__class__.__name__:
            if child.K1 is not None:
                k1_max = max_sv(child.K1)
                k3_max = max_sv(child.K3)
                if cur_max < max(k1_max, k3_max):
                    cur_max = max(k1_max, k3_max)
        else:
            fut_max = max_conv_lip(child, cur_max)
            if fut_max > cur_max:
                cur_max = fut_max
    return cur_max


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--data-dir", default="./cifar-data", type=str)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument(
        "--lr-schedule", default="multistep", choices=["cyclic", "multistep"]
    )
    parser.add_argument("--lr-min", default=0.0, type=float)
    parser.add_argument("--lr-max", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--epsilon", default=36, type=int)
    parser.add_argument("--in-channels", default=16, type=int)
    parser.add_argument("--block-size", default=4, type=int, help="model type")
    parser.add_argument(
        "--out-dir", default="lipnet", type=str, help="Output directory"
    )
    parser.add_argument("--seed", default=3407, type=int, help="Random seed")
    parser.add_argument(
        "--opt-level",
        default="O2",
        type=str,
        choices=["O0", "O1", "O2"],
        help="O0 is FP32 training, O1 is Mixed Precision, "
        'and O2 is "Almost FP16" Mixed Precision',
    )
    parser.add_argument(
        "--conv-type",
        default="standard",
        type=str,
        choices=["standard", "bcop", "skew", "tt", "sott"],
        help="standard, skew symmetric, bcop convolution or " "tt decomposition",
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["cifar10", "cifar100"],
        help="dataset to use for training",
    )
    parser.add_argument(
        "--loss-scale",
        default="1.0",
        type=str,
        choices=["1.0", "dynamic"],
        help='If loss_scale is "dynamic", adaptively adjust '
        "the loss scale over time",
    )
    parser.add_argument(
        "--dec-rank",
        default=0,
        type=str,
        help="TT decomposition rank, not applied if 0",
    )
    parser.add_argument(
        "--assigning-freq",
        default=0,
        type=int,
        help="Assigning frequency, not applied if 0",
    )
    parser.add_argument(
        "--clipping", default=None, type=str, choices=[None, "clip", "assign"]
    )
    parser.add_argument("--clip_to", default=1, type=float)
    parser.add_argument("--orthogonal-k", default=1, type=float)
    return parser.parse_args()


def main():
    args = get_args()

    if args.dataset == "cifar10":
        args.in_channels = 32
    elif args.dataset == "cifar100":
        args.in_channels = 32
    else:
        raise Exception("Unknown dataset ", args.dataset)

    args.out_dir = (
        args.out_dir
        + "_"
        + str(args.dataset)
        + "_"
        + str(args.block_size)
        + "_"
        + str(args.conv_type)
    )
    args.block_size = int(args.block_size)

    os.makedirs(args.out_dir, exist_ok=True)
    logfile = os.path.join(args.out_dir, "output.log")
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=os.path.join(args.out_dir, "output.log"),
    )
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, args.dataset
    )
    std = cifar10_std
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    else:
        raise Exception("Unknown dataset")

    # Evaluation at early stopping
    model_kwargs = {}
    if args.dec_rank != 0:
        dec_rank = list(map(int, args.dec_rank.split(", ")))
        if len(dec_rank) == 1:
            dec_rank = dec_rank[0]
        model_kwargs["decomposition_rank"] = dec_rank
    model = LipNet_n(
        args.conv_type,
        in_channels=args.in_channels,
        num_blocks=args.block_size,
        num_classes=num_classes,
        **model_kwargs,
    ).to(device)
    model.train()
    print(model)
    print(sum(p.numel() for p in model.parameters()))

    if args.conv_type == "skew":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=args.lr_max,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        opt = torch.optim.SGD(
            model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=0.0
        )

    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=args.lr_min,
            max_lr=args.lr_max,
            step_size_up=lr_steps / 20,
            step_size_down=(3 * lr_steps) / 20,
        )
    elif args.lr_schedule == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[lr_steps // 2, (3 * lr_steps) // 4], gamma=0.1
        )
    else:
        raise ValueError("No scheduler!")

    # Training
    std = torch.tensor(std).to(device)
    L = 1 / torch.max(std)
    prev_test_acc = 0.0
    start_train_time = time.time()
    wandb.init(project="singular_values")
    logger.info(
        "Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t Train "
        "Robust \t Train Cert \t Test Loss \t Test Acc \t Test "
        "Robust \t Test Cert"
    )
    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            X, y = X.to(device), y.to(device)
            output = model(X)

            ce_loss = criterion(output, y)
            loss = ce_loss
            if args.conv_type == "sott":
                ort_loss, shapes = orthogonal_loss(model)
                loss += (args.orthogonal_k / shapes) * ort_loss
                wandb.log(
                    {"ort_loss_item": ort_loss / shapes, "ce_loss_item": ce_loss.item()}
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                if args.assigning_freq != 0 and i != 0 and i % args.assigning_freq == 0:
                    print(i)
                    constrain_conv(
                        model, mode=args.clipping, clip_assign_value=args.clip_to
                    )
            curr_correct = output.max(1)[1] == y
            train_loss += ce_loss.item() * y.size(0)
            train_acc += curr_correct.sum().item()
            train_n += y.size(0)
            scheduler.step()

        _, _, mean_train_cert, train_robust_acc = evaluate_certificates(
            train_loader, model, L
        )

        last_state_dict = copy.deepcopy(model.state_dict())

        # Check current test accuracy of model
        test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(
            test_loader, model, L
        )
        if robust_acc > prev_test_acc:
            model_path = os.path.join(args.out_dir, "best.pth")
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, model_path)
            prev_test_acc = robust_acc
            best_epoch = epoch

        with torch.no_grad():
            max_sv_now = max_conv_lip(model)

        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info(
            "%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t "
            "%.4f \t %.4f \t %.4f \t %.4f",
            epoch,
            epoch_time - start_epoch_time,
            lr,
            train_loss / train_n,
            train_acc / train_n,
            train_robust_acc,
            mean_train_cert,
            test_loss,
            test_acc,
            robust_acc,
            mean_cert,
        )
        wandb.log(
            {
                "loss": train_loss / train_n,
                "val_loss": test_loss,
                "acc": train_acc / train_n,
                "val_acc": test_acc,
                "epoch": epoch,
                "lr": lr,
                "train_robust": train_robust_acc,
                "train_cert": mean_train_cert,
                "max_sv": max_sv_now,
                "time": epoch_time - start_epoch_time,
                "test_robust": robust_acc,
                "test_cert": mean_cert,
            }
        )

        model_path = os.path.join(args.out_dir, "last.pth")
        torch.save(last_state_dict, model_path)

        trainer_state_dict = {"epoch": epoch, "optimizer_state_dict": opt.state_dict()}
        opt_path = os.path.join(args.out_dir, "last_opt.pth")
        torch.save(trainer_state_dict, opt_path)

    if args.conv_type == "skew":
        sigma_array = test_real_sn(model)
        s_min, s_mean, s_max = sigma_array.min(), sigma_array.mean(), sigma_array.max()
        logger.info("Real sigma statistics: %.4f \t %.4f \t %.4f", s_min, s_mean, s_max)

    train_time = time.time()

    logger.info("Total train time: %.4f minutes", (train_time - start_train_time) / 60)

    checkpoints_dir = pathlib.Path("/home/checkpoints")
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    torch.save(
        {
            "epoch": epoch,
            "best_model_state_dict": best_state_dict,
            "last_model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "loss": loss,
        },
        checkpoints_dir / f"lipconv_{args.conv_type}_"
        f"rank_{args.dec_rank}_clipping_{args.clipping}_"
        f"clip_to_{args.clip_to}_epochs_{args.epochs}_"
        f"opt_SGD_init_lr_{args.lr_max}_"
        f"batch_size_{args.batch_size}_"
        f"blocksize_{args.block_size}.pth",
    )

    # Evaluation at early stopping
    model_test = LipNet_n(
        args.conv_type,
        in_channels=args.in_channels,
        num_blocks=args.block_size,
        num_classes=num_classes,
        **model_kwargs,
    ).to(device)
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(
        test_loader, model_test, L
    )
    total_time = time.time() - start_test_time

    logger.info(
        "Best Epoch \t Test Loss \t Test Acc \t Robust Acc \t Mean " "Cert \t Test Time"
    )
    logger.info(
        "%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f",
        best_epoch,
        test_loss,
        test_acc,
        robust_acc,
        mean_cert,
        total_time,
    )

    # Evaluation at last model
    model_test.load_state_dict(last_state_dict)
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(
        test_loader, model_test, L
    )
    total_time = time.time() - start_test_time

    logger.info(
        "Last Epoch \t Test Loss \t Test Acc \t Robust Acc \t Mean " "Cert \t Test Time"
    )
    logger.info(
        "%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f",
        epoch,
        test_loss,
        test_acc,
        robust_acc,
        mean_cert,
        total_time,
    )


if __name__ == "__main__":
    main()
