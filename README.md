# ViTSoup: A Once-for-All Deployment Paradigm for Efficient Vision Transformers using Coprime-Window Supernets

## Overview

We propose a once-for-all deployment paradigm designed to automatically generate a family of efficient vision Transformers that can adapt to various visual tasks. Our method involves three key stages: supernet pre-training, supernet fine-tuning on downstream tasks, and efficient subnet search under deployment constraints. We specifically propose a pre-trained supernet, ViTSoup, founded on a large-scale architecture space that employs a novel coprime-size window mechanism to facilitate cross-window information exchange without the need for complex shifting operations. When evaluated on ImageNet-1K classification, a discovered subnet achieves a top-1 accuracy of 81.8\% with 35.8 million parameters and 5.9 GFLOPs. Furthermore, when fine-tuned for image segmentation, our method yields efficient subnets that attain 50.6 PQ on COCO panoptic, 42.2 AP on COCO instance, and 47.0 mIoU on ADE20K semantic segmentation—demonstrating competitive performance with significantly fewer parameters and lower computational cost. This work establishes a flexible and cost-effective framework for adapting large-scale vision models to diverse tasks and deployment scenarios, bridging the gap between high-capacity pre-training and efficient, task-specific inference.

![The overview of our paradigm](https://github.com/kunjing96/ViTSoup/blob/master/figs/OVERVIEW4.png)
![The overview of ViTSoup](https://github.com/kunjing96/ViTSoup/blob/master/figs/OVERVIEW.png)
![Weight entanglement](https://github.com/kunjing96/ViTSoup/blob/master/figs/WEIGHTENTANGLEMENT.png)
![Shifted window-based attention](https://github.com/kunjing96/ViTSoup/blob/master/figs/SWA.png)
![Coprime-size window-based attention](https://github.com/kunjing96/ViTSoup/blob/master/figs/CWA.png)
![Image segmentation head](https://github.com/kunjing96/ViTSoup/blob/master/figs/SEGHEAD1.png)
![Multi-scale deformable Transformer block](https://github.com/kunjing96/ViTSoup/blob/master/figs/SEGHEAD2.png)
![Mask Transformer block](https://github.com/kunjing96/ViTSoup/blob/master/figs/SEGHEAD3.png)


## Requirements and Dependencies
All requirements and dependencies refer to these two repositories: SwinTransformer](https://github.com/microsoft/Swin-Transformer) and [Mask2Former](https://github.com/facebookresearch/Mask2Former).

## Reproducing Experiments

Pretrain supernet for image classification
```bash
cd ViTSoup
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/swin/super_swin_patch4_window7_224_no_shift_tiny.yaml --data-path <imagenet-path> --batch-size 128 
```

Search subnet for image classification
```bash
cd ViTSoup
python evolution.py --cfg configs/swin/swin_small_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128 
```

Finetune supernet for image segmentation
```bash
cd ViTSoup4Seg
python train_net.py --num-gpus 8 --config-file configs/xxx/xxx/super/maskformer2_R50_bs16_50ep.yaml
```

Search subnet for image segmentation
```bash
cd ViTSoup4Seg
python evolution.py --cfg configs/xxx/xxx/super/maskformer2_R50_bs16_50ep.yaml
```

Evaluate subnet for image segmentation
```bash
cd ViTSoup4Seg
python evolution.py --cfg configs/xxx/xxx/subnet/maskformer2_R50_bs16_50ep.yaml  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## Manuscript Affiliation
This repository contains the official implementation for the manuscript submitted to [*The Visual Computer*](https://link.springer.com/journal/371).
