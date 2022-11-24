import torch
from torch.nn import Sequential
from torch.nn import Linear, BatchNorm2d, ReLU, Flatten, Conv2d, MaxPool2d


def vgg19(in_chan=3, num_classes=10, **kwargs):
    sizes = [
        64,
        64,
        -1,
        128,
        128,
        -1,
        256,
        256,
        256,
        256,
        -1,
        512,
        512,
        512,
        512,
        -1,
        512,
        512,
        512,
        512,
        -1,
    ]

    return VGG(in_chan, num_classes, sizes)


class VGG(torch.nn.Module):
    def __init__(self, in_chan, num_classes, sizes):
        super().__init__()

        model = []

        for s in sizes:
            if s == -1:
                # Pooling
                model.append(MaxPool2d((2, 2)))
            else:
                # Conv layer
                model.append(Conv2d(in_chan, s, 3, padding=1))

                model.append(BatchNorm2d(s, eps=0.001))
                model.append(ReLU())
                in_chan = s

        model.append(Flatten())
        model.append(Linear(512, 512))
        model.append(ReLU())
        model.append(Linear(512, 512))
        model.append(ReLU())
        model.append(Linear(512, num_classes))

        self.model = Sequential(*model)

        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
