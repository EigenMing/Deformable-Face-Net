# Deformable-Face-Net
This is the code for the paper "Deformable Face Net: Learning Pose Invariant Feature with Pose Aware Feature Alignment for Face Recognition"

## Introduction
The Deformable Face Net (DFN) is designed for pose invariant face recognition (PIFR).

- The deformable convolution module attempts to simultaneously learn face recognition oriented alignment and identity-preserving feature extraction.
- The displacement consistency loss (DCL) enforces the learnt displacement fields for aligning faces to be locally consistent.
- The identity consistency loss (ICL) minimizes the intra-class feature variation.
- The DCL and ICL loss functions jointly learn pose-aware displacement fields for deformable convolutions in the DFN. 

The DFN achieves quite promising performance with relatively light network structure, especially for large poses.

## Requirements
- Install [MXNet](https://github.com/apache/incubator-mxnet) on a machine with CUDA GPU

## Training Data Preparation
All face images need to be aligned with 5 points landmark detection and cropped to 256x256 using the [SeetaFace2 Engine](https://github.com/seetaface/SeetaFaceEngine2). Besides, the random mirror and the random crop with size 248x248 are recommended.

- Training the DFN-L with the DCL and ICL loss functions requires pair-wise training samples. For example, when preparing a batch with N images, the 1th image and the (N/2+1)th image should be two faces from the same identity and the 2th image and the (N/2+2)th image should be two faces from another identity.

## How to Use
This repository contains the implementation of the DFN-L structure, the DCL and ICL loss functions. For instance, you can get their mxnet symbol by:
```python
from DFN_L_DCL_ICL.sym_DFN_L_DCL_ICL import get_symbol
sym=get_symbol(num_classes)
```

The num_classes should be the number of classess in your training data.
## To Be Continue

## Citation
If you find this work useful in your research, please cite:

[1] Mingjie He, Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen, "Deformable Face Net: Learning Pose Invariant Feature with Pose Aware Feature Alignment for Face Recognition," 14th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2019), Lille, France, May 14-18, 2019. 