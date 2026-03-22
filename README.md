# BACFormer
Code for paper "BACFormer: a robust boundary-aware transformer for medical image segmentation".
Our paper is currently under review by the Knowledge-Based Systems journal. Please stay tuned!

## 1. Environment

- Please prepare an environment with Ubuntu 20.04, with Python 3.9.7, PyTorch 1.13.0, and CUDA 11.7

## 2. Train/Test

- Train

```bash
python train.py --dataset LeftAtrium --root_path your DATA_DIR --max_epochs 400 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 12
```

- Test 

```bash
python test.py --dataset LeftAtrium --is_savenii --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 400 --base_lr 0.05 --img_size 224 --batch_size 24
```

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)