#!/bin/bash

#Note: "trained_on_imagenet_004000.t7" can be found on "firefly" at "/home/altrogge/code/27_06_2016/faster-rcnn.torch/" !!!

#th perform_ROIPoolingMassa_singleImg.lua -cfg config/imagenet.lua -name imagenet1 -train ILSVRC2015_DET.t7 -resultDir logs6

#th perform_ROIPoolingMassa_singleImg.lua -cfg config/imagenet.lua -name imagenet1 -train ILSVRC2015_DET.t7 -resultDir logs6 -restore trained_on_imagenet_004000.t7

th perform_ROIPoolingMassa.lua -cfg config/imagenet.lua -name imagenet1 -train ILSVRC2015_DET.t7 -resultDir logs6 -restore trained_on_imagenet_004000.t7
