# CCNet-Pure-Pytorch
Criss-Cross Attention for Semantic Segmentation in pure Pytorch with faster and more precise implementation.
## Introduction
I unofficially re-implement [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721) in pure Pytorch for better compatibility under different versions and environments. Many previous open-source projects employ a Cuda extension for Pytorch, which suffer from problems of compatibility and precision loss. Moreover, Cuda extension may not be optimized and accelerated by Pytorch, when we set cudnn.benchmark = True. To address these issues, I design a Criss-Cross Attention operation in our [CC.py] based on tensor transformation in Pytorch, which is implemented in parallel and shows a faster speed and more precise in forward result and backward gradient.
## My Operation and Performances
Previous Criss-Cross Attention projects are using a Cuda extension for Pytorch. Here I design a more elegant pure Pytorch implementation for Criss-Cross Attention in [CC.py]. To check the correctness and compare with CUDA [cc_attention](https://github.com/speedinghzl/CCNet) of the official one, run the [check.py].

To check the correctness, I check my pure pytorch CC() and the official CUDA CrissCross(), the inputs are Q, K and V, respectively.<br>
![Input](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/1.PNG)
