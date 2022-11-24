from .SOC.utils import get_loaders, evaluate_certificates, cifar10_std
import argparse
from os import listdir
from os.path import isfile, join
from .train import arch_mapping, conv2tt
from .SOTT import SOTT
import torch
import datetime
from .SOC.robust_net import LipNet_n
from .SOC.skew_symmetric_conv import SOC


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chp-folder", type=str)
    return parser.parse_args()


def load_checkpoint(path, info):  # last_model_state_dict
    filename = path.split("/")[-1]
    num_classes = 10
    if "dataset_cifar100" in filename:
        num_classes = 100
    if info[-1].endswith(".pth"):
        info[-1] = info[-1][:-4]
    model_name = info[0]
    print(path)
    if info[0] == "lipconv":
        model_kwargs = {}
        if info[1] == "sott":
            model_kwargs["decomposition_rank"] = int(info[3])
        for i in range(len(info)):
            if info[i] == "blocksize":
                blocksize = info[i + 1]
                if blocksize.endswith(".pth"):
                    blocksize = blocksize[:-4]
                blocksize = int(blocksize)
                model_kwargs["num_blocks"] = blocksize
        model = LipNet_n(info[1], **model_kwargs).to(device)
    else:
        model = arch_mapping[model_name](num_classes=num_classes).to(device)
        if "tt" in filename:
            rank = int(info[3])
            dec_scale = -1
            no_compress_first = False
            for i in range(len(info)):
                if info[i] == "ds":
                    dec_scale = float(info[i + 1])
                if info[i] == "no-compress-first":
                    no_compress_first = False if info[i + 1] == "False" else True
            if "sott" in filename:
                conv2tt(
                    model,
                    new_layer=SOTT,
                    decomposition_rank=rank,
                    decomposition_scale=dec_scale,
                    device=device,
                )
            else:
                conv2tt(
                    model,
                    decomposition_rank=rank,
                    decomposition_scale=dec_scale,
                    device=device,
                    no_compress_first=no_compress_first,
                )
        elif "skew" in filename:
            conv2tt(model, new_layer=SOC, device=device)
    checkpoint = torch.load(path, map_location=device)
    if "loss" in checkpoint.keys():
        print("loss:", checkpoint["loss"])
    if info[0] == "lipconv":
        if "last_model_state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["last_model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    return model


def main():
    args = get_args()
    std = cifar10_std
    std = torch.tensor(std).to(device)
    L = 1 / torch.max(std)
    train_loader, test_loader = get_loaders("./cifar-data", 128, "cifar10")
    onlyfiles = [
        f for f in listdir(args.chp_folder) if isfile(join(args.chp_folder, f))
    ]
    header_string = "filename\ttest_loss\ttest_acc\tmean_cert\trobust_acc"
    log_filename = (
        f"/home/provable_robust_acc_for_{len(onlyfiles)}_chps_"
        f"{datetime.datetime.now()}.txt"
    )
    file_w = open(log_filename, "w")
    print(header_string)
    print(header_string, file=file_w)
    for i in range(len(onlyfiles)):
        file = join(args.chp_folder, onlyfiles[i])
        model = load_checkpoint(file, onlyfiles[i].split("_"))
        test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(
            test_loader, model, L
        )
        result_string = (
            f"{file}\t{test_loss.item()}\t{test_acc.item()}\t"
            f"{mean_cert}\t{robust_acc}"
        )
        print(result_string)
        print(result_string, file=file_w)
    file_w.close()


if __name__ == "__main__":
    main()
