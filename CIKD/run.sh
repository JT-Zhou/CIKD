#res32x4_wrn16_2 res110_8x4
# nohup python train.py --cfg configs/cifar100/dkd/vgg13_vgg8.yaml &
python train.py --cfg configs/cifar100/kd.yaml
python train.py --cfg configs/kd_M.yaml