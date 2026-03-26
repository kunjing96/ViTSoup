# ViTSoup: A Once-for-All Deployment Paradigm for Efficient Vision Transformers using Coprime-Window Supernets

## Overview

We propose a once-for-all deployment paradigm designed to automatically generate a family of efficient vision Transformers that can adapt to various visual tasks. Our method involves three key stages: supernet pre-training, supernet fine-tuning on downstream tasks, and efficient subnet search under deployment constraints. We specifically propose a pre-trained supernet, ViTSoup, founded on a large-scale architecture space that employs a novel coprime-size window mechanism to facilitate cross-window information exchange without the need for complex shifting operations. When evaluated on ImageNet-1K classification, a discovered subnet achieves a top-1 accuracy of 81.8\% with 35.8 million parameters and 5.9 GFLOPs. Furthermore, when fine-tuned for image segmentation, our method yields efficient subnets that attain 50.6 PQ on COCO panoptic, 42.2 AP on COCO instance, and 47.0 mIoU on ADE20K semantic segmentation—demonstrating competitive performance with significantly fewer parameters and lower computational cost. This work establishes a flexible and cost-effective framework for adapting large-scale vision models to diverse tasks and deployment scenarios, bridging the gap between high-capacity pre-training and efficient, task-specific inference.

[The overview of our paradigm](https://github.com/user-attachments/files/26257296/OVERVIEW4.pdf)
[The overview of ViTSoup](https://github.com/user-attachments/files/26257293/OVERVIEW.pdf)
[Weight entanglement](https://github.com/user-attachments/files/26257304/WEIGHTENTANGLEMENT.pdf)
[Shifted window-based attention](https://github.com/user-attachments/files/26257303/SWA.pdf)
[Coprime-size window-based attention](https://github.com/user-attachments/files/26257302/CWA.pdf)
[Image segmentation head](https://github.com/user-attachments/files/26257298/SEGHEAD1.pdf)
[Multi-scale deformable Transformer block](https://github.com/user-attachments/files/26257300/SEGHEAD2.pdf)
[Mask Transformer block](https://github.com/user-attachments/files/26257301/SEGHEAD3.pdf)


## Requirements and Dependencies
All requirements and dependencies refer to these two repositories: [AutoFormer](https://github.com/ICCV2021/Autoformer) and [Mask2Former](https://github.com/facebookresearch/Mask2Former).

## Reproducing Experiments
TODO

## Manuscript Affiliation
This repository contains the official implementation for the manuscript submitted to [*The Visual Computer*](https://link.springer.com/journal/371).
