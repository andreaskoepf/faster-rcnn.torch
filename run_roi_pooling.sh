#!/bin/bash

#th perform_ROIPoolingMassa_singleImg.lua -cfg config/imagenet.lua -name imagenet1 -train ILSVRC2015_DET.t7 -resultDir logs6

#th perform_ROIPoolingMassa_singleImg.lua -cfg config/imagenet.lua -name imagenet1 -train ILSVRC2015_DET.t7 -resultDir logs6 -restore trained_on_imagenet_004000.t7

th perform_ROIPoolingMassa.lua -cfg config/imagenet.lua -name imagenet1 -train ILSVRC2015_DET.t7 -resultDir logs6 -restore trained_on_imagenet_004000.t7
