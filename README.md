# 2D3D-MATR: 2D-3D Matching Transformer for Detection-free Registration between Images and Point Clouds

PyTorch implementation of the paper:

[2D3D-MATR: 2D-3D Matching Transformer for Detection-free Registration between Images and Point Clouds](https://arxiv.org/abs/2308.05667).

Minhao Li, [Zheng Qin](https://scholar.google.com/citations?user=DnHBAN0AAAAJ), [Zhirui Gao](https://scholar.google.com/citations?user=IqtwGzYAAAAJ), [Renjiao Yi](https://renjiaoyi.github.io), [Chengyang Zhu](https://scholar.google.com/citations?user=vThu1hIAAAAJ), [Yulan Guo](https://scholar.google.com/citations?user=WQRNvdsAAAAJ), and [Kai Xu](https://scholar.google.com/citations?user=GuVkg-8AAAAJ).

## Introduction

The commonly adopted detect-then-match approach to registration finds difficulties in the cross-modality cases due to the incompatible keypoint detection and inconsistent feature description. We propose, 2D3D-MATR, a detection-free method for accurate and robust registration between images and point clouds. Our method adopts a coarse-to-fine pipeline where it first computes coarse correspondences between downsampled patches of the input image and the point cloud and then extends them to form dense correspondences between pixels and points within the patch region. The coarse-level patch matching is based on transformer which jointly learns global contextual constraints with self-attention and cross-modality correlations with cross-attention. To resolve the scale ambiguity in patch matching, we construct a multi-scale pyramid for each image patch and learn to find for each point patch the best matching image patch at a proper resolution level. Extensive experiments on two public benchmarks demonstrate that 2D3D-MATR outperforms the previous state-of-the-art P2-Net by around $20$ percentage points on inlier ratio and over $10$ points on registration recall.

![](assets/teaser.png)

## News

2023.10.08: Code and pretrained models on 7Scenes and RGB-D Scenes V2 released.

2023.08.10: Paper is available at [arXiv](https://arxiv.org/abs/2203.05667).

2023.07.14: This work is accepted by ICCV 2023.

## Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n matr2d3d python==3.8
conda activate matr2d3d

# 2. Install vision3d following https://github.com/qinzheng93/vision3d
```

The code has been tested on Python 3.8, PyTorch 1.13.1, Ubuntu 22.04, GCC 11.3 and CUDA 11.7, but it should work with other configurations.

## Pre-trained Weights

We provide pre-trained weights from [BaiduYun](https://pan.baidu.com/s/1-HXn9xayYNTJJa-Fjgy4JA)(extraction code: 34ks). Please download the latest weights and put them in `weights` directory.

## 7Scenes

### Data preparation

The dataset can be downloaded from [BaiduYun](https://pan.baidu.com/s/1duymPG4dJte4Yx-qov5yeg)(extraction code: m7mc). The data should be organized as follows:

```text
--data--7Scenes--metadata
              |--data--chess
                    |--fire
                    |--heads
                    |--office
                    |--pumpkin
                    |--redkitchen
                    |--stairs
```

### Training

The code for 7Scenes is in `experiments/2d3dmatr.7scenes.stage4.level3.stage1`. Use the following command for training.

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Testing

Use the following command for testing.

```bash
CUDA_VISIBLE_DEVICES=0 ./eval.sh EPOCH
```

`EPOCH` is the epoch id.

We also provide pretrained weights in `weights`, use the following command to test the pretrained weights.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint=/path/to/2D3DMATR/weights/2d3dmatr-7scenes.pth
CUDA_VISIBLE_DEVICES=0 python eval.py --test_epoch=-1
```

## RGB-D Scenes V2

### Data preparation

The dataset can be downloaded from [BaiduYun](https://pan.baidu.com/s/14A2y8jghCdk6nAZa0_yEZA)(extraction code: 2dc7). The data should be organized as follows:

```text
--data--RGBDScenesV2--metadata
                   |--data--rgbd-scenes-v2-scene_01
                         |--...
                         |--rgbd-scenes-v2-scene_14
```

### Training

The code for RGB-D Scenes V2 is in `experiments/2d3dmatr.rgbdv2.stage4.level3.stage1`. Use the following command for training.

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Testing

Use the following command for testing.

```bash
CUDA_VISIBLE_DEVICES=0 ./eval.sh EPOCH
```

`EPOCH` is the epoch id.

We also provide pretrained weights in `weights`, use the following command to test the pretrained weights.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint=/path/to/2D3DMATR/weights/2d3dmatr-rgbdv2.pth
CUDA_VISIBLE_DEVICES=0 python eval.py --test_epoch=-1
```

## Results

We evaluate GeoTransformer on the 7Scenes and RGB-D Scenes V2 benchmarks.

| Benchmark |  FMR  |  IR   |  RR   |
| :-------- | :---: | :---: | :---: |
| 7Scenes   | 92.1  | 50.1  | 75.8  |
| RGB-D V2  | 90.8  | 32.4  | 56.4  |

NOTE: the results could be a little different due to the randomness of RANSAC-PnP.

## Citation

```bibtex
@inproceedings{li20232d3d,
  title={2D3D-MATR: 2D-3D Matching Transformer for Detection-free Registration between Images and Point Clouds},
  author={Li, Minhao and Qin, Zheng and Gao, Zhirui and Yi, Renjiao and Zhu, Chenyang and Guo, Yulan and Xu, Kai},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14128--14138},
  year={2023}
}
```

## Acknowledgements

- [D2-Net](https://github.com/mihaidusmanu/d2-net)
- [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch)
- [RPMNet](https://github.com/yewzijian/RPMNet)
- [vision3d](https://github.com/qinzheng93/vision3d)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
