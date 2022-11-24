import argparse
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from autoattack import AutoAttack
from robustbench.data import _load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from tqdm import tqdm
from .measure_provable_robust import load_checkpoint

from .utils import get_testloader, create_cifar10_testset, create_cifar100_testset
from importlib.resources import read_text
import wget
import tarfile
from pathlib import Path


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def load_txt(path: str) -> list:
    return [line.rstrip("\n") for line in open(path)]


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1)  # top-k index: size (B, k)
        pred = pred.t()  # size (k, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            acc.append(correct_k * 100.0 / batch_size)

        if len(acc) == 1:
            return acc[0]
        else:
            return acc


def create_barplot(accs: dict, title: str, savepath: str):
    y = list(accs.values())
    x = np.arange(len(y))
    xticks_ = list(accs.keys())
    xticks = []
    for i in range(len(xticks_)):
        if "avg" in xticks_[i]:
            xticks.append(xticks_[i])
        else:
            xticks.append(xticks_[i][:-4])

    plt.bar(x, y)
    for i, j in zip(x, y):
        plt.text(i, j, f"{j:.1f}", ha="center", va="bottom", fontsize=7)

    plt.title(title)
    plt.ylabel("Accuracy (%)")

    plt.ylim(0, 100)

    plt.xticks(x, xticks, rotation=90)
    plt.yticks(np.linspace(0, 100, 11))

    plt.subplots_adjust(bottom=0.36, top=0.92)
    plt.grid(axis="y")
    plt.savefig(savepath)
    plt.close()


def get_fname(weight_path: str):
    return ".".join(weight_path.split("/")[-1].split(".")[:-1])


class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root: str, name: str, transform=None, target_transform=None):
        super(CIFAR10C, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        data_path = os.path.join(root, name)
        target_path = os.path.join(root, "labels.npy")

        self.data = np.load(data_path, allow_pickle=True)
        self.targets = np.load(target_path, allow_pickle=True)

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset: int, random_subset: bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)


def get_accs(
    cp,
    cifarc,
    autoattacks,
    dataset_root,
    cifar_c_res,
    autoattacks_res,
    cifar_c_root,
    dataset_name,
    gouk_transforms,
):
    model = load_checkpoint(cp, cp.split("/")[-1].split("_"))

    if cifarc:
        if dataset_name == "cifar10":
            cifar_mean, cifar_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            cifar_mean, cifar_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        if gouk_transforms:
            cifar_mean, cifar_std = 0.5, 0.5
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(cifar_mean, cifar_std),
            ]
        )

        corruptions = read_text(__package__, "corruptions.txt")
        corruptions = [i[:-1] for i in corruptions.split("\n")]

        cifar_folder = Path(cifar_c_root)
        if not cifar_folder.exists():
            if dataset_name == "cifar10":
                url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
            else:
                url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
            filename = wget.download(url)
            mytar = tarfile.open(filename)
            mytar.extractall(".")
            mytar.close()

        accs = dict()
        with tqdm(total=len(corruptions), ncols=80) as pbar:
            for ci, cname in enumerate(os.listdir(cifar_c_root)):
                if cname == "labels.npy":
                    continue
                dataset = CIFAR10C(cifar_c_root, cname, transform=test_transform)
                loader = DataLoader(
                    dataset, batch_size=128, shuffle=False, num_workers=4
                )

                acc_meter = AverageMeter()
                with torch.no_grad():
                    for itr, (x, y) in enumerate(loader):
                        x = x.to(device)
                        y = y.to(device)

                        z = model(x)
                        acc, _ = accuracy(z, y, topk=(1, 5))
                        acc_meter.update(acc.item())

                accs[f"{cname}"] = acc_meter.avg

                pbar.set_postfix_str(f"{cname}: {acc_meter.avg:.2f}")
                pbar.update()

        avg = np.mean(list(accs.values()))
        accs["avg"] = avg

        print("Cifar-C:", avg)

        os.makedirs(cifar_c_res, exist_ok=True)
        save_name = os.path.join(cifar_c_res, cp[:-4].split("/")[-1])
        create_barplot(accs, f"avg={avg:.2f}", save_name + ".png")

    if autoattacks:
        if dataset_name == "cifar10":
            testset = create_cifar10_testset(
                dataset_root, gouk_transforms=gouk_transforms
            )
        else:
            testset = create_cifar100_testset(
                dataset_root, gouk_transforms=gouk_transforms
            )
        x_test, y_test = _load_dataset(testset)

        os.makedirs(autoattacks_res, exist_ok=True)

        adversary = AutoAttack(
            model.to(device),
            norm="L2",
            eps=1 / 2,
            version="standard",
            log_path=os.path.join(autoattacks_res, cp[:-4].split("/")[-1] + ".txt"),
        )
        adversary.run_standard_evaluation(x_test.to(device), y_test.to(device))


if __name__ == "__main__":
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    SEED = 3407
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ALL_METRICS = ["ece", "autoattacks", "cifar-c"]

    """### Parsing arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS)
    parser.add_argument("--cifar-c-root", default="CIFAR-10-C", type=str)
    parser.add_argument("--dataset-root", default="./data", type=str)
    parser.add_argument("--cifar-c-res", default="cifar-c-res", type=str)
    parser.add_argument("--autoattacks-res", default="autoattacks-res", type=str)
    parser.add_argument(
        "--gouk-transforms", dest="gouk_transforms", action="store_true"
    )

    parser.set_defaults(
        nesterov=False,
        lip_bn=False,
        collect_sing_vals=False,
        clip_linear=False,
        no_compress_first=False,
        gouk_transforms=False,
    )

    args = parser.parse_args()

    for elem in args.metrics:
        assert elem in ALL_METRICS

    cps = glob.glob(f"{args.checkpoints_dir}/*.pth")
    if len(cps) == 0:
        print("Oh no, the folder for checkpoints is empty!")
    else:
        cps.sort()

        # ECE

        if "ece" in args.metrics:
            for cp in cps:
                cp = cp.replace("\\\\", "/")
                cp = cp.replace("\\", "/")
                model = load_checkpoint(cp, cp.split("/")[-1].split("_"))

                file_name = cp.split("\\")[-1]
                dataset = "cifar10"
                if "dataset_cifar100" in file_name:
                    dataset = "cifar100"
                print(f"\n\nWorking on\n\t{file_name}")

                testloader = get_testloader(
                    args.batch_size,
                    args.dataset_root,
                    dataset,
                    gouk_transforms=args.gouk_transforms,
                )

                logits_list = []
                labels_list = []
                with torch.no_grad():
                    for input, label in testloader:
                        input = input.to(device)
                        logits = model(input)
                        logits_list.append(logits)
                        labels_list.append(label)
                    logits = torch.cat(logits_list).to(device)
                    labels = torch.cat(labels_list).to(device)

                ece_criterion = _ECELoss().to(device)
                met = ece_criterion(logits, labels).item()
                print(f"ECE: {met}")

        # Cifar-C & AutoAttacks
        if "cifar-c" in args.metrics or "autoattacks" in args.metrics:
            for cp in cps:
                file_name = cp.split("/")[-1]
                dataset = "cifar10"
                if "dataset_cifar100" in file_name:
                    dataset = "cifar100"
                print(f"\n\nWorking on\n\t{file_name}")
                get_accs(
                    cp,
                    "cifar-c" in args.metrics,
                    "autoattacks" in args.metrics,
                    args.dataset_root,
                    args.cifar_c_res,
                    args.autoattacks_res,
                    args.cifar_c_root,
                    dataset,
                    gouk_transforms=args.gouk_transforms,
                )
