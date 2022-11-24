import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2507, 0.2507, 0.2507)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).to(device)
std = torch.tensor(cifar10_std).view(3, 1, 1).to(device)

upper_limit = (1 - mu) / std
lower_limit = (0 - mu) / std


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size, dataset_name="cifar10"):
    if dataset_name == "cifar10":
        dataset_func = datasets.CIFAR10
    elif dataset_name == "cifar100":
        dataset_func = datasets.CIFAR100

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    num_workers = 4
    train_dataset = dataset_func(
        dir_, train=True, transform=train_transform, download=True
    )
    test_dataset = dataset_func(
        dir_, train=False, transform=test_transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def evaluate_certificates(test_loader, model, L, epsilon=36.0):
    losses_list = []
    certificates_list = []
    correct_list = []
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y, reduction="none")
            losses_list.append(loss)

            output_max, output_amax = torch.max(output, dim=1)

            onehot = torch.zeros_like(output).to(device)
            onehot[torch.arange(output.shape[0]), output_amax] = 1.0

            output_trunc = output - onehot * 1e6

            output_nextmax = torch.max(output_trunc, dim=1)[0]
            output_diff = output_max - output_nextmax

            certificates = output_diff / (math.sqrt(2) * L)
            correct = output_amax == y

            certificates_list.append(certificates)
            correct_list.append(correct)

        losses_array = torch.cat(losses_list, dim=0).cpu().numpy()
        certificates_array = torch.cat(certificates_list, dim=0).cpu().numpy()
        correct_array = torch.cat(correct_list, dim=0).cpu().numpy()

    mean_loss = np.mean(losses_array)
    mean_acc = np.mean(correct_array)

    mean_certificates = (certificates_array * correct_array).sum() / correct_array.sum()

    robust_correct_array = (certificates_array > (epsilon / 255.0)) & correct_array
    robust_correct = robust_correct_array.sum() / robust_correct_array.shape[0]
    return mean_loss, mean_acc, mean_certificates, robust_correct
