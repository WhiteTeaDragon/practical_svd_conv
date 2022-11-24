# WRN-16-10 baseline training examples
## Cifar10 examples
python -m practical_svd_conv.src.train --dataset cifar10 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer standard --batch-size 128 --nesterov --weight-dec 0.0001 --checkpoints-path checkpoints --gouk-transforms
python -m practical_svd_conv.src.train --dataset cifar10 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer tt --dec-rank 192 --batch-size 128 --orthogonal-k 100000 --nesterov --weight-dec 0.0001 --checkpoints-path checkpoints --gouk-transforms

## Cifar100 examples
python -m practical_svd_conv.src.train --dataset cifar100 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer standard --batch-size 128 --nesterov --weight-dec 0.0001 --checkpoints-path checkpoints --gouk-transforms
python -m practical_svd_conv.src.train --dataset cifar100 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer tt --dec-rank 192 --batch-size 128 --orthogonal-k 100000 --nesterov --weight-dec 0.0001 --checkpoints-path checkpoints --gouk-transforms

# WRN-16-10 clipping example
python -m practical_svd_conv.src.train --dataset cifar100 --architecture wrn16-10 --epochs 200 --clipping clip --clip_to 2 --clip_freq 100 --init-lr 0.1 --opt SGD --new-layer tt --dec-rank 320 --orthogonal-k 100000 --batch-size 128 --nesterov --weight-dec 0.0001 --checkpoints-path checkpoints --gouk-transforms

# WRN-16-10 division examples
## Cifar10
python -m practical_svd_conv.src.train --dataset cifar10 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer tt --dec-rank 192 --orthogonal-k 100000 --batch-size 128 --clipping divide-by-largest --clip_freq 1 --clip_to 7 --lip-bn --freq-bn 1 --bn-eps 10 --clip-linear --clip_linear_to 7 --nesterov --weight-dec 0.0001 --checkpoints-path checkpoints --gouk-transforms

## Cifar100
python -m practical_svd_conv.src.train --dataset cifar100 --architecture wrn16-10 --epochs 200 --init-lr 0.1 --opt SGD --new-layer tt --dec-rank 320 --orthogonal-k 100000 --batch-size 128 --clipping divide-by-largest --clip_freq 1 --clip_to 10 --lip-bn --freq-bn 1 --bn-eps 6.1 --clip-linear --clip_linear_to 3.9 --nesterov --weight-dec 0.00005 --checkpoints-path checkpoints --gouk-transforms

# VGG baseline training examples
## Cifar10 examples
python -m practical_svd_conv.src.train --checkpoints-path checkpoints --architecture vgg19 --init-lr 0.0001 --opt Adam --batch-size 128 --epochs 140 --weight-dec 0
python -m practical_svd_conv.src.train --checkpoints-path checkpoints --architecture vgg19 --init-lr 0.0001 --opt Adam --batch-size 128 --weight-dec 0 --new-layer tt --dec-rank 192 --orthogonal-k 100000 --epochs 140

## Cifar100 examples
python -m practical_svd_conv.src.train --dataset cifar100 --checkpoints-path checkpoints --architecture vgg19 --init-lr 0.0001 --opt Adam --batch-size 128 --epochs 140 --weight-dec 0
python -m practical_svd_conv.src.train --dataset cifar100 --checkpoints-path checkpoints --architecture vgg19 --init-lr 0.0001 --opt Adam --batch-size 128 --weight-dec 0 --new-layer tt --dec-rank 192 --orthogonal-k 100000 --epochs 140

# VGG clipping example
python -m practical_svd_conv.src.train --checkpoints-path checkpoints --architecture vgg19 --init-lr 0.0001 --opt Adam --batch-size 128 --weight-dec 0 --new-layer tt --dec-rank 256 --orthogonal-k 100000 --epochs 140 --clipping clip --clip_freq 100 --clip_to 0.5

# VGG division example
## Cifar10
python -m practical_svd_conv.src.train --architecture vgg19 --epochs 140 --init-lr 0.0001 --opt Adam --new-layer standard --batch-size 128 --clipping divide-by-largest --clip_freq 1 --clip_to 1.5 --lip-bn --freq-bn 1 --bn-eps 8 --clip-linear --clip_linear_to 1.5 --nesterov --weight-dec 0 --checkpoints-path checkpoints

## Cifar100 (division params are the same)
python -m practical_svd_conv.src.train --architecture vgg19 --epochs 140 --init-lr 0.0001 --opt Adam --new-layer standard --batch-size 128 --clipping divide-by-largest --clip_freq 1 --clip_to 1.5 --lip-bn --freq-bn 1 --bn-eps 8 --clip-linear --clip_linear_to 1.5 --nesterov --weight-dec 0 --checkpoints-path checkpoints --dataset cifar100

# LipConv
python -m practical_svd_conv.src.SOC.train_robust --block-size 1 --lr-max 0.1  --conv-type sott --dec-rank 128 --orthogonal-k 5000
python -m practical_svd_conv.src.SOC.train_robust --block-size 1 --lr-max 0.1  --conv-type sott --dec-rank 256 --orthogonal-k 5000
python -m practical_svd_conv.src.SOC.train_robust --block-size 4 --lr-max 0.05  --conv-type sott --dec-rank 128 --orthogonal-k 70000
python -m practical_svd_conv.src.SOC.train_robust --block-size 4 --lr-max 0.05  --conv-type sott --dec-rank 256 --orthogonal-k 70000
python -m practical_svd_conv.src.SOC.train_robust --block-size 6 --lr-max 0.05  --conv-type sott --dec-rank 128 --orthogonal-k 200000
python -m practical_svd_conv.src.SOC.train_robust --block-size 6 --lr-max 0.05  --conv-type sott --dec-rank 256 --orthogonal-k 200000

# Robust metrics
python -m practical_svd_conv.src.robust_metrics --checkpoints-dir /home/checkpoints --gouk-transforms