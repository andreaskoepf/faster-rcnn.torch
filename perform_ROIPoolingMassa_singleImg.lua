----------------------------------------------------------
-- Only for one single input image:
-- Prepare proposal net output for ROIPooling 
-- (with help of module "AnchorOutputToROIPoolingInput")
-- and then perform ROIPooling 
-- (with help of "inn"-package module "ROIPooling")
-- (or with help of ROIPoolingMassa.lua by fmassa)
----------------------------------------------------------
require 'cunn'
require 'utilities'
require 'BatchIterator'
require 'AnchorOutputToROIPoolingInput'
require 'ROIPoolingMassa'
--local inn = require 'inn'


-- command line options
cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Perform ROIPooling on the proposal output of a trained proposal net.')
cmd:text()
cmd:text('=== Training ===')
cmd:option('-cfg', 'config/imagenet.lua', 'configuration file')
cmd:option('-model', 'models/vgg_small.lua', 'model factory file')
cmd:option('-name', 'imgnet', 'experiment name, snapshot prefix')
cmd:option('-train', 'ILSVRC2015_DET.t7', 'training data file name')
cmd:option('-restore', '', 'network snapshot file name to load')
cmd:option('-resultDir', 'logs', 'Folder for storing all result. (training process ect)')
cmd:text('=== Misc ===')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-gpuid', 0, 'device ID (CUDA), (use -1 for CPU)')
cmd:option('-seed', 0, 'random seed (0 = no fixed seed)')
print('Command line args:')
local opt = cmd:parse(arg or {})
print(opt)
print('Options:')
local cfg = dofile(opt.cfg)
print(cfg)
os.execute(('mkdir -p %s'):format(opt.resultDir))


function load_model(cfg, model_path, network_filename, cuda)
  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg)
  graph.dot(model.pnet.fg, 'pnet',string.format('%s/pnet_fg',opt.resultDir))
  graph.dot(model.pnet.bg, 'pnet',string.format('%s/pnet_bg',opt.resultDir))
  if cuda then
    model.cnet:cuda()
    model.pnet:cuda()
  end
  -- combine parameters from pnet and cnet into flat tensors
  local weights, gradient = combine_and_flatten_parameters(model.pnet, model.cnet)
  local training_stats
  if network_filename and #network_filename > 0 then
    local stored = load_obj(network_filename)
    training_stats = stored.stats
    weights:copy(stored.weights)
  end
  return model, weights, gradient, training_stats
end


-- Main part:
-------------
local training_data = load_obj(opt.train)
local model, weights, gradient, training_stats = load_model(cfg, opt.model, opt.restore, true)
if not training_stats then
  training_stats = { pcls={}, preg={}, dcls={}, dreg={} }
end
local batch_iterator = BatchIterator.new(model, training_data)
local batch = batch_iterator:nextTraining()

local ancOutToROIPoolIn = nn.AnchorOutputToROIPoolingInput.new(model)
local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
--local roiPooling = inn.ROIPooling(kw,kh):cuda()
local roiPooling = nn.ROIPoolingMassa(kw,kh):cuda()
-- self.spatial_scale = 1
-- roiPooling:setSpatialScale(1/16)

for i,x in ipairs(batch) do
  local img = x.img:cuda()

  -- run forward convolution
  local outputs = model.pnet:forward(img)

  print('Start of bboxTable calculation.')
  local bboxTable = {}
  bboxTable = ancOutToROIPoolIn:forward(outputs)
  print('#bboxTable:')
  print(#bboxTable)

  if #bboxTable > 0 then

    -- inn.ROIPooling and also ROIPooling by fmassa is Spatial Adaptive Max Pooling (amp) Layer 
    -- for region proposals used in FastRCNN.
    -- Both expect a table on input, i.e. input = {data, rois}:
    --  * First argument data = input[1] is features in N x D x H x W, where
    --     ** N = #images,
    --     ** D = dim = #feature maps,
    --     ** H = height, 
    --     ** W = width
    --    (i.e. 1 x 384 x 14 x 22)
    --  * Second argument rois = input[2] is bounding boxes in B x 5, where
    --     ** B = #bboxes,
    --     ** 5 = {imgId, minX, minY, maxX, maxY} in feature map coordinates
    -- see e.g.: https://github.com/szagoruyko/imagine-nn

    local roiPoolingInput1 = torch.CudaTensor(1, 
                                              outputs[5]:size()[1], 
                                              outputs[5]:size()[2], 
                                              outputs[5]:size()[3])
    -- torch.CudaTensor of size 1 x 384 x e.g. 14 x e.g. 19 (NxDxHxW = #img x dim x height x width)
    roiPoolingInput1[1] = outputs[5]
    print('roiPoolingInput1:size():')
    print(roiPoolingInput1:size())
    print('roiPoolingInput1:type():')
    print(roiPoolingInput1:type())

    local roiPoolingInput2 = torch.CudaTensor(#bboxTable, 5)
    local imageId = 1
    local minX, minY, maxX, maxY = 0, 0, 0, 0
    for i=1,#bboxTable do
      --imageId = bboxTable[i][1]
      minX = bboxTable[i][1]
      minY = bboxTable[i][2]
      maxX = bboxTable[i][3]
      maxY = bboxTable[i][4]
      roiPoolingInput2[{i,{}}] = torch.CudaTensor({imageId, minX, minY, maxX, maxY})
    end
    --roiPoolingInput2[{i,{}}] = bboxTable[i]
    print('roiPoolingInput2:')
    print(roiPoolingInput2) -- torch.CudaTensor of size #bboxes x 5 (Bx5 = #bboxes x {imgId, minX, minY, maxX, maxY})

    local inputTable = {roiPoolingInput1, roiPoolingInput2}
    print('inputTable:')
    print(inputTable)

    local roiPoolingOutput = roiPooling:forward(inputTable)
    -- torch.CudaTensor of size #bboxes x 384 x 6 x 6  (B x D x kw x kh = #bboxes x dim x kw x kh)
    print('roiPoolingOutput:size():')
    print(roiPoolingOutput:size())
    print('roiPoolingOutput:type():')
    print(roiPoolingOutput:type())

    --local roiPoolingOutputPlane = roiPoolingOutput:view(kh * kw * cnet_input_planes)

    os.exit()
  end --if #bboxTable > 0 then
end --for i,x in ipairs(batch) do
