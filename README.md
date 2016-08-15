# faster-rcnn
This is an experimental Torch7 implementation of Faster RCNN - a convnet for object detection with a region proposal network.
For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1504.08083v1.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

## Work in progress
Status: Basic detection in my personal environment works.
A 'small' network is used that can be trained on a 4 GB GPU with 800x450 images.
Began experimenting with ImageNet: create-imagenet-traindat.lua can be used to create a training data file for the ILSVRC2015 dataset.

## Notes:
- The number of "scales" in the config file (e.g. config/imagenet.lua) and the number of "anchor nets" in the model (e.g. models/vgg_small.lua) have to be the same!
- To enable training with only the first batch of images, set "self.training.i = 1" at the beginning of "function BatchIterator:nextTraining(count)" in "BatchIterator.lua". Training with only one image batch might be helpful for testing some network components/functionalities, because, when training on always the same image batch, the training error should go down very soon (otherwise something's wrong).

## Todo:
- [!] regularly evaluate net during traning to compute test-set loss
- generate training/test graph with [gnuplot](https://github.com/torch/gnuplot)
- add final per class non-maximum suppression to generate final proposals (already included but eval code rewrite still pending)
- remove hard coded path, create full set of command line options
- add parameters to separately enable/disable training of bounding box proposal-network and fine-tuning + classification.

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
