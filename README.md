# Understanding Loss Landscapes of Neural Network Models in Solving Partial Differential Equations


This repository contains the PyTorch code for the paper

> Keke Wu, Rui Du, Jingrun Chen and Xiang Zhou. Understanding Loss Landscapes of Neural Network Models in Solving Partial Differential Equations. [[PDF on arXiv]](https://arxiv.org/abs/2103.11069)  

## Abstract

Solving partial differential equations (PDEs) by parametrizing its solution by neural networks (NNs) has been popular in the past a few years. However, different types of loss functions can be proposed for the same PDE. For the Poisson equation, the loss function can be based on the weak formulation of energy variation or the least squares method, which leads to the deep Ritz model and deep Galerkin model, respectively. But loss landscapes from these different models give arise to different practical performance of training the NN parameters. To investigate and understand such practical differences, we propose to compare the loss landscapes of these models, which are both high dimensional and highly non-convex. In such settings, the roughness is more important than the traditional eigenvalue analysis to describe the non-convexity. We contribute to the landscape comparisons by proposing a roughness index to scientifically and quantitatively describe the heuristic concept of "roughness" of landscape around minimizers. This index is based on random projections and the variance of (normalized) total variation for one dimensional projected functions, and it is efficient to compute. A large roughness index hints an oscillatory landscape profile as a severe challenge for the first order optimization method. We apply this index to the two models for the Poisson equation and our empirical results reveal a consistent general observation that the landscapes from the deep Galerkin method around its local minimizers are less rough than the deep Ritz method, which supports the observed gain in accuracy of the deep Galerkin method. 

## Setup

Environment: 
One or more multi-GPU node(s) with the following main software/libraries installed:


- Python 3.6.5
- PyTorch 1.7.0
- CUDA 10.2
- numpy 1.14.3
- matplotlib 2.2.2
- ...

Remark: One may have some problem with PyTorch about data on cuda and cpu when computing the roughness index, you can move the data and network on cpu, i.e., delete "torch.cuda.set_device(0)" and all ".cuda()".

## Citation

If you find this code useful in your research, please cite:

```
@article{understanding,
title={Understanding Loss Landscapes of Neural Network Models in Solving Partial Differential Equations},
author={Wu, Keke and Du, Rui and Chen, Jingrun and Zhou, Xiang},
journal={arXiv preprint arXiv:2103.11069},
year={2021}
}

```

## 
