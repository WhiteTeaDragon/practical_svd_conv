# Towards Practical Control of Singular Values of Convolutional Layers

This repository is the official implementation of our NeurIPS 2022 paper "Towards Practical Control of Singular Values of Convolutional Layers" by Alexandra Senderovich, Ekaterina Bulatova, Anton Obukhov and Maxim Rakhuba [[OpenReview]](http://openreview.net/forum?id=T5TtjbhlAZH).

It demonstrates how to perform low-rank neural network reparameterization to speed up the control over singular values of convolutional layers. The code provides all experiments (LipConv and WideResNet-16-10) from the paper.

## Installation

In order to install all the necessary dependencies run the following command:

```
!pip install -r requirements.txt
```

In case of problems with generic requirements, fall back to 
[requirements_reproducibility.txt](requirements_reproducibility.txt).

## Logging

The training code performs logging to [Weights and Biases](wandb.ai). Upon the first run, please enter your wandb credentials, which can be obtained by registering a free account 
with the service.

The [code](src/robust_metrics.py) for computing robust metrics logs results only locally. Computation of ECE takes up less than a minute, and its results are logged only to the console. The accuracy on Cifar-C dataset and the accuracy after applying AutoAttack are logged to the respective paths, specified by command line arguments.

## Training

### Datasets
We use Cifar-10 and Cifar-100 datasets. In our code they are downloaded via creating instances of respective torchvision classes (e.g. ```torchvision.datasets.CIFAR10```). If you already have one of the datasets on your machine in the suitable format for these classes, then you might specify the path to it via ```--dataset-root``` argument.

### WideResNet-16-10

To run WideResNet-16-10 baseline training on Cifar10 or Cifar100, execute one the following commands:

```shell
python -m practical_svd_conv.src.train --dataset cifar10 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer standard --batch-size 128 --nesterov --weight-dec 0.0001 --checkpoints-path <path to checkpoints> --gouk-transforms
python -m practical_svd_conv.src.train --dataset cifar100 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer standard --batch-size 128 --nesterov --weight-dec 0.0001 --checkpoints-path <path to checkpoints> --gouk-transforms
```

To use TT-decomposition, one should change the value of the ```--new-layer``` parameter to ```tt``` and vary the value of the ```--dec-rank``` parameter. Moreover, to achieve the accuracy from the paper it is important to use orthogonal loss by specifying the parameter ```--orthogonal-k```. For example, to train WideResNet-16-10 with rank 192 the following command should be executed:

```shell
python -m practical_svd_conv.src.train --dataset cifar10 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer tt --dec-rank 192 --batch-size 128 --orthogonal-k 100000 --nesterov --weight-dec 0.0001 --checkpoints-path <path to checkpoints> --gouk-transforms
```

In order to run experiments with clipping, three new arguments should be added to previous commands:
* ```--clipping clip ```, which distinguished clipping and division operations
* ```--clip_to X```, where X is the largest possible singular value after clipping operation
* ```--clip_freq Y```, which indicates that clipping operation is done every Y training steps (100 in all our experiments)

Example of running training with clipping:

```shell
python -m practical_svd_conv.src.train --dataset cifar100 --architecture wrn16-10 --epochs 200 --clipping clip --clip_to 2 --clip_freq 100 --init-lr 0.1 --opt SGD --new-layer tt --dec-rank 320 --orthogonal-k 100000 --batch-size 128 --nesterov --weight-dec 0.0001 --checkpoints-path <path to checkpoints> --gouk-transforms
```

To run experiments with division, even more arguments should be added:
* ```--clipping divide_by_largest```
* ```--clip_to X``` and ```--clip_freq Y```, where X is the constraint on singular values of convolutional layers and Y is the frequency of applying division operation (equals to 1 in all our experiments)
* ```--lip_bn```, which turns on the constraint on batch normalization layers
* ```--bn-eps X``` and ```--freq-bn Y```, which are similar to ```--clip_to``` and ```--clip_freq```, but for batch normalization layers
* ```--clip-linear```, which turns on the constraint on linear layers 
* ```--clip_linear_to X```, where X is the constraint on singular values of linear layers, and frequency is the same as for the convolutional layers

In our experiments all these parameters did not differ for different ranks and were taken from the [repository](https://github.com/henrygouk/keras-lipschitz-networks) of the corresponding article by Gouk et al. Specifically, we used these two sets of parameters for Cifar10 and Cifar100:
```shell
python -m practical_svd_conv.src.train --dataset cifar10 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer tt --dec-rank 192 --orthogonal-k 100000 --batch-size 128 --clipping divide-by-largest --clip_freq 1 --clip_to 7 --lip-bn --freq-bn 1 --bn-eps 10 --clip-linear --clip_linear_to 7 --nesterov --weight-dec 0.0001 --checkpoints-path <path to checkpoints> --gouk-transforms
python -m practical_svd_conv.src.train --dataset cifar100 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer tt --dec-rank 320 --orthogonal-k 100000 --batch-size 128 --clipping divide-by-largest --clip_freq 1 --clip_to 10 --lip-bn --freq-bn 1 --bn-eps 6.1 --clip-linear --clip_linear_to 3.9 --nesterov --weight-dec 0.00005 --checkpoints-path <path to checkpoints> --gouk-transforms
```

### VGG
Commands for training baselines:
```shell
python -m practical_svd_conv.src.train --checkpoints-path <path to checkpoints> --architecture vgg19 --init-lr 0.0001 --opt Adam --batch-size 128 --epochs 140 --weight-dec 0
python -m practical_svd_conv.src.train --dataset cifar100 --checkpoints-path <path to checkpoints> --architecture vgg19 --init-lr 0.0001 --opt Adam --batch-size 128 --epochs 140 --weight-dec 0
```

Clipping example:
```shell
python -m practical_svd_conv.src.train --checkpoints-path <path to checkpoints> --architecture vgg19 --init-lr 0.0001 --opt Adam --batch-size 128 --weight-dec 0 --new-layer tt --dec-rank 256 --orthogonal-k 100000 --epochs 140 --clipping clip --clip_freq 100 --clip_to 0.5
```

Division example (the parameters for Cifar100 are the same as for Cifar10, because they gave a better result than those from Gouk):
```shell
python -m practical_svd_conv.src.train --architecture vgg19 --epochs 140 --init-lr 0.0001 --opt Adam --new-layer standard --batch-size 128 --clipping divide-by-largest --clip_freq 1 --clip_to 1.5 --lip-bn --freq-bn 1 --bn-eps 8 --clip-linear --clip_linear_to 1.5 --nesterov --weight-dec 0 --checkpoints-path <path to checkpoints>
python -m practical_svd_conv.src.train --architecture vgg19 --epochs 140 --init-lr 0.0001 --opt Adam --new-layer standard --batch-size 128 --clipping divide-by-largest --clip_freq 1 --clip_to 1.5 --lip-bn --freq-bn 1 --bn-eps 8 --clip-linear --clip_linear_to 1.5 --nesterov --weight-dec 0 --checkpoints-path <path to checkpoints> --dataset cifar100
```

### LipConv
Here are all the commands that were run to obtain the results presented in the paper:
```shell
python -m practical_svd_conv.src.SOC.train_robust --block-size 1 --lr-max 0.1  --conv-type sott --dec-rank 128 --orthogonal-k 5000
python -m practical_svd_conv.src.SOC.train_robust --block-size 1 --lr-max 0.1  --conv-type sott --dec-rank 256 --orthogonal-k 5000
python -m practical_svd_conv.src.SOC.train_robust --block-size 4 --lr-max 0.05  --conv-type sott --dec-rank 128 --orthogonal-k 70000
python -m practical_svd_conv.src.SOC.train_robust --block-size 4 --lr-max 0.05  --conv-type sott --dec-rank 256 --orthogonal-k 70000
python -m practical_svd_conv.src.SOC.train_robust --block-size 6 --lr-max 0.05  --conv-type sott --dec-rank 128 --orthogonal-k 200000
python -m practical_svd_conv.src.SOC.train_robust --block-size 6 --lr-max 0.05  --conv-type sott --dec-rank 256 --orthogonal-k 200000
```

## Robust metrics computation
To get ECE, accuracy on Cifar-C and after applying AutoAttack for model checkpoints one should run the following command:
```shell
python -m practical_svd_conv.src.robust_metrics --checkpoints-dir <path to folder with checkpoints> --gouk-transforms
```

In our code the Cifar-C dataset, corresponding to the dataset a checkpoint was trained on, is downloaded automatically to the path specified by ```--cifar-c-root``` argument. The path to the regular, uncorrupted dataset can be regulated by ```--dataset-root```.

Moreover, let us note that the file "corruptions.txt" is essential for running Cifar-C evaluation.