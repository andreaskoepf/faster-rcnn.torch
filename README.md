# faster-rcnn
This is an experimental Torch7 implementation of Faster RCNN - a convnet for object detection with a region proposal network.
For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1504.08083v1.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

## [!] WARNING: Work in progress
Status: Basic detection in my personal environment works.
A 'small' network is used that can be trained on a 4 GB GPU with 400x450 images.

## Todo:
- [!] regularly evaluate net during traning to compute test-set loss
- generate training graph with [gnuplot](https://github.com/torch/gnuplot)
- add final per class non-maximum suppression to generate final proposals
- remove hard coded path, create full set of command line options
- add parameters to separately enable/disable training of bounding box proposal-network and fine-tuning + classification.
- change image loading strategy to handle images of different sizes (currently everything is resized to 800x450)
- use single image in different scales during training
- add loader-code for image-net data ([ILSVRC2015](http://image-net.org/challenges/LSVRC/2015/index))
- create fast ROI-pooling layer for torch
- verify: computation of positions on conv-output based on bounding box proposals

## Experiments to run:
- test smaller networks
- 6x6 vs. 7x7 classification ROI-pooling output size
- impact of RGB, YUV, Lab color space
- test relevance of local contrast normalization

## References / Review / Useful Links
* [SPP Paper](http://arxiv.org/pdf/1406.4729.pdf)
* [Fast R-CNN paper](http://arxiv.org/abs/1504.08083)
* [R-CNN paper](http://arxiv.org/abs/1311.2524)
* [vgg net, cifar.torch](https://github.com/szagoruyko/cifar.torch/blob/master/models/vgg_bn_drop.lua)
* [55 epoche learn rate schedule](
https://github.com/soumith/imagenet-multiGPU.torch/blob/master/train.lua)