# CCNet-Pure-Pytorch
Criss-Cross Attention (2d&3d) for Semantic Segmentation in pure Pytorch with a faster and more precise implementation.
### Updates

****2021/03: Three kinds of pure-pytorch implementation of 3D CCNet Module is released in [CC3d.py](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/networks/CC3d.py). And you can check their correctness in [check3dby2d.py](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/cc_attention/check3dby2d.py) and [check3d.py](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/cc_attention/check3d.py)
## Introduction
I unofficially re-implement [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721) in pure Pytorch for better compatibility under different versions and environments. Many previous open-source projects employ a Cuda extension for Pytorch, which suffer from problems of compatibility and precision loss. Moreover, Cuda extension may not be optimized and accelerated by Pytorch, when we set cudnn.benchmark = True. To address these issues, I design a Criss-Cross Attention operation in our [CC.py](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/networks/CC.py) based on tensor transformation in Pytorch, which is implemented in parallel and shows a faster speed and more precise in the forward result and backward gradient.
## My Operation and Performances
CUDA extension is not necessary.
Previous Criss-Cross Attention projects are using a Cuda extension for Pytorch. Here I design a more elegant pure Pytorch implementation for Criss-Cross Attention in [CC.py](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/networks/CC.py). To check the correctness and compare it with CUDA [cc_attention](https://github.com/speedinghzl/CCNet) of the official one, run the [check.py](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/cc_attention/check.py).

To check the correctness, I check my pure pytorch CC() and the official CUDA CrissCross(), the inputs are Query, Key and Value, respectively.<br><br>
![Input](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/1.PNG)<br>
The theoretical output should be 3. The output of our CC() is <br><br>
![CC](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/21.PNG)<br>
But the output of official CUDA CrissCross() is not exactly 3<br><br>
![CUDA](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/3.PNG)<br>
Then I check the gradient, the theoretical gradient of z is 1. Gradient of CC() is excatly 1, but gradient of CUDA CrissCross() is 0.9999998212. <br><br>
![g](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/4.PNG)<br>
As for the speed of tranning and testing, I compare my Pytorch Criss-Cross Attention and the official CUDA Criss-Cross Attention in this project. For batch size 4 at 4 2080Ti with Ohem,  my Pytorch Criss-Cross Attention costs 14m32s, and the official CUDA Criss-Cross Attention costs 15m22s on Cityscapes trainning set. For evaluation with batch size 1 at 1 2080Ti using single scale, my Pytorch Criss-Cross Attention costs 28m44s, and the official CUDA Criss-Cross Attention costs 30m59s on Cityscapes val set.<br>
Evaluatations for a same CKPT in single scale by my pure pytorch implementation and official cc_attention.<br>
My module<br>
![em](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/mycc.PNG)<br>
Official CUDA cc_attention<br>
![ec](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/Fig/rcuda.PNG)<br>
Our pure Pytorch implementation ([CC.py](https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/networks/CC.py)) is faster and more precise, as well as more compatibale.
### SynBN
For better compatibility under different versions and environments, I decide to use pure Pytorch implementation without using  Cuda [inplace-abn](https://github.com/mapillary/inplace_abn). I adopt [Synchronized-BatchNorm-PyTorch
](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch), so it costs more GPU memory than inplace-abn. And I will try to realize an efficient inplace-abn in the future.<br>
## Requirements
CUDA extension is not necessary.<br>
If you do not want to compare our implementation with the Cuda version, you just need Python 3, Pytorch 1.2 or 0.4, OpenCV and PIL.<br>
If you want to compare our implementation with the Cuda version, you need Pytorch 1.1 or 1.2 and apex.
```bash
# Install **Pytorch-1.1**
$ conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# Install **Apex**
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
## Training and Evaluation
ImageNet Pre-trained Model can be downloaded from [resnet101-imagenet.pth](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth).
```bash
$ export CUDA_VISIBLE_DEVICES=0,1,2,3
$ python train.py --data-dir /data/datasets/Cityscapes/ --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu 0,1,2,3 --learning-rate 0.01 --input-size 769,769 --weight-decay 0.0001 --batch-size 4 --num-steps 60000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model ccnet


$ python evaluate.py --data-dir /data/datasets/Cityscapes/  --recurrence 2 --model ccnet --restore-from ./snapshots/CS_scenes_60000.pth --whole Ture --gpu 0 --recurrence 2 --batch-size 1

```
## Dataset
I implement this on [CityScapes](https://www.cityscapes-dataset.com/) dataset.
## Thanks to the Third Party Libs
[CCNet: Criss-Cross Attention for Semantic Segmentation](https://github.com/speedinghzl/CCNet)<br>
[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

## To do

- [ ] Implement Inplace-abn in pure Pytorch.
