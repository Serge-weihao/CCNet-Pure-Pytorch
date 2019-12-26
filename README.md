# CCNet-Pure-Pytorch
Criss-Cross Attention for Semantic Segmentation in pure Pytorch with faster and more precise implementation.
## Introduction
I unofficially re-implement [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721) in pure Pytorch for better compatibility under different versions and environments. Many previous open-source projects employ a Cuda extension for Pytorch, which suffer from problems of compatibility and precision loss. Moreover, Cuda extension may not be optimized and accelerated by Pytorch, when we set cudnn.benchmark = True. To address these issues, I design a Criss-Cross Attention operation in our [CC.py] based on tensor transformation in Pytorch, which is implemented in parallel and shows a faster speed and more precise in the forward result and backward gradient.
## My Operation and Performances
Previous Criss-Cross Attention projects are using a Cuda extension for Pytorch. Here I design a more elegant pure Pytorch implementation for Criss-Cross Attention in [CC.py]. To check the correctness and compare it with CUDA [cc_attention](https://github.com/speedinghzl/CCNet) of the official one, run the [check.py].

To check the correctness, I check my pure pytorch CC() and the official CUDA CrissCross(), the inputs are Q, K and V, respectively.<br><br>
![Input](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/1.PNG)<br>
The theoretical output should be 3. The output of our CC() is <br><br>
![CC](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/21.PNG)<br>
But the output of official CUDA CrissCross() is not exactly 3<br><br>
![CUDA](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/3.PNG)<br>
Then I check the gradient, the theoretical gradient of z is 1. Gradient of CC() is excatly 1, but gradient of CUDA CrissCross() is 0.9999998212. <br><br>
![g](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/4.PNG)<br>
As for the speed of tranning and testing, I compare my Pytorch Criss-Cross Attention and the official CUDA Criss-Cross Attention in this project. For batch size 4 at 4 2080Ti with Ohem,  my Pytorch Criss-Cross Attention costs 14m32s, and the official CUDA Criss-Cross Attention costs 15m22s on Cityscapes trainning set. For evaluation with batch size 1 at 1 2080Ti using single scale, my Pytorch Criss-Cross Attention costs 28m44s, and the official CUDA Criss-Cross Attention costs 30m59s on Cityscapes val set.<br>
### SynBN
For better compatibility under different versions and environments, I decide to use pure Pytorch implementation
