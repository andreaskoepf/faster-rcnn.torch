# faster-rcnn
This is an experimental Torch7 implementation of Faster RCNN - a convnet for object detection with a region proposal network.
For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

## Get data set:

### ImageNet

TODO

### Pascal Voc 2007

1. Download the training, validation, test data and VOCdevkit

  ```Shell
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
  ```

2. Extract all of these tars into one directory named `VOCdevkit`

  ```Shell
  tar xvf VOCtrainval_06-Nov-2007.tar
  tar xvf VOCtest_06-Nov-2007.tar
  tar xvf VOCdevkit_08-Jun-2007.tar
  ```

3. It should have this basic structure

  ```Shell
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
    # ... and several other directories ...
    ```

4. Create symlinks for the PASCAL VOC dataset

  ```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. Run script:
    '''Shell
    th create-pascal_voc-traindata.lua
    '''
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012

### COCO

TODO

## Start Training:

### Proposal network (Pnet)
Start with a pretrained vgg16 network. (Get network from: https://github.com/szagoruyko/fastrcnn-models.torch)

1. Using only the first Batch for training
'''
th main.lua -cfg config/pascal_voc.lua -model models/pretrained/vgg16_ori.lua -name pascal_voc -train /data/VOCdevkit/VOC2007/pascal_voc_2007.t7 -plot 50 -opti adam -lr 1e-2 -gpuid 0 -resultDir log_onlyPnet_allImgs -mode onlyPnet -oneBatchTraining
 '''
1. Using full dataset for training
'''
th main.lua -cfg config/pascal_voc.lua -model models/pretrained/vgg16_ori.lua -name pascal_voc -train /data/VOCdevkit/VOC2007/pascal_voc_2007.t7 -plot 50 -opti adam -lr 1e-5 -gpuid 0 -resultDir log_onlyPnet_allImgs -mode onlyPnet
 '''

### Classification network (Cnet)

After the pnet is trained you can start training the Cnet by:

1. Using only the first Batch for training
'''
th main.lua -cfg config/pascal_voc.lua -model models/pretrained/vgg16_ori.lua -name pascal_voc -train /data/VOCdevkit/VOC2007/pascal_voc_2007.t7 -plot 50 -opti adam -lr 1e-2 -gpuid 0 -resultDir log_onlyPnet_allImgs -mode onlyCnet -oneBatchTraining
 '''
1. Using full dataset for training
'''
th main.lua -cfg config/pascal_voc.lua -model models/pretrained/vgg16_ori.lua -name pascal_voc -train /data/VOCdevkit/VOC2007/pascal_voc_2007.t7 -plot 50 -opti adam -lr 1e-5 -gpuid 0 -restorePnet log_onlyPnet_allImgs/pascal_voc_050000.t7 -resultDir log_onlyPnet_allImgs -mode onlyCnet
 '''

## Todo:

### Detector
- add final per class non-maximum suppression to generate final proposals (already included but eval code rewrite still pending)

### Training
- add joint training to merge Pnet and Cnet to one network

### Docu
extend readme file

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
