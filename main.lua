require 'torch'
require 'pl'
require 'optim'
require 'image'
require 'nngraph'
require 'cunn'
require 'nms'

require 'utilities'
require 'model'
require 'Anchors'
require 'BatchIterator'
require 'Objective'

-- parameters

local base_path = '/home/koepf/datasets/brickset_all/'
local testset_path = '/home/koepf/datasets/realbricks/'

 local duplo_cfg = {
  class_count = 16,  -- excluding background class
  target_smaller_side = 450,
  --scales = { 48, 96, 192, 384 },
  scales = { 32, 64, 128, 256 },
  max_pixel_size = 1000,
  --normalization = { method = 'contrastive', width = 7 },
  normalization = { method = 'none' },
  color_space = 'yuv',
  roi_pooling = { kw = 6, kh = 6 },
  examples_base_path = '/home/koepf/datasets/brickset_all/',
  background_base_path = '/home/koepf/datasets/background/',
  batch_size = 256,
  positive_threshold = 0.5, 
  negative_threshold = 0.25,
  best_match = true,
  nearby_aversion = true
}

 local imgnet_cfg = {
  class_count = 200,  -- excluding background class
  --target_smaller_side = 600,
  --scales = { 64, 128, 256, 512 },
  target_smaller_side = 480,
  scales = { 48, 96, 192, 384 },
  max_pixel_size = 1000,
  normalization = { method = 'none' },
  color_space = 'yuv',
  roi_pooling = { kw = 6, kh = 6 },
  examples_base_path = '',
  background_base_path = '',
  batch_size = 256,
  positive_threshold = 0.6, 
  negative_threshold = 0.3,
  best_match = true,
  nearby_aversion = true
}

-- command line options
cmd = torch.CmdLine()
cmd:addTime()

cmd:text()
cmd:text('Training a convnet for region proposals')
cmd:text()

cmd:text('=== Training ===')
cmd:option('-lr', 1E-4, 'learn rate')
cmd:option('-rms_decay', 0.9, 'RMSprop moving average dissolving factor')
cmd:option('-opti', 'rmsprop', 'Optimizer')

cmd:text('=== Misc ===')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-gpuid', 0, 'device ID (CUDA), (use -1 for CPU)')
cmd:option('-seed', 0, 'random seed (0 = no fixed seed)')

local opt = cmd:parse(arg or {})
print(opt)

-- system configuration
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpuid + 1)  -- nvidia tools start counting at 0
torch.setnumthreads(opt.threads)
if opt.seed ~= 0 then
  torch.manualSeed(opt.seed)
  cutorch.manualSeed(opt.seed)
end

function load_image_auto_size(fn, target_smaller_side, max_pixel_size, color_space)
  local img = image.load(path.join(base_path, fn), 3, 'float')
  local dim = img:size()
  
  local w, h
  if dim[2] < dim[3] then
    -- height is smaller than width, set h to target_size
    w = math.min(dim[3] * target_smaller_side/dim[2], max_pixel_size)
    h = dim[2] * w/dim[3]
  else
    -- width is smaller than height, set w to target_size
    h = math.min(dim[2] * target_smaller_side/dim[1], max_pixel_size)
    w = dim[3] * h/dim[2]
  end
  
  img = image.scale(img, w, h)
  
  if color_space == 'yuv' then
    img = image.rgb2yuv(img)
  elseif color_space == 'lab' then
    img = image.rgb2lab(img)
  elseif color_space == 'hsv' then
    img = image.rgb2hsv(img)
  end

  return img, dim
end

function graph_evaluate(cfg, training_data_filename, network_filename, normalize, bgclass)
  -- TODO: REWRITE
  local training_data = load_obj(training_data_filename)
  local ground_truth = training_data.ground_truth
  local image_file_names = training_data.validation_set

  local stored = load_obj(network_filename)
  local pnet = create_proposal_net()
  local cnet = create_classifaction_net(cfg.roi_pooling.kw, cfg.roi_pooling.kh, 300, cfg.class_count + 1)
  pnet:cuda()
  cnet:cuda()
  
  local anchors = Anchors.new(pnet, cfg.scales)
  local localizer = Localizer.new(pnet.outnode.children[5])
  
  -- restore weights
  local weights, gradient = combine_and_flatten_parameters(pnet, cnet)
  weights:copy(stored.weights)
  
  local red = torch.Tensor({1,0,0})
  local green = torch.Tensor({0,1,0})
  local blue = torch.Tensor({0,0,1})
  local white = torch.Tensor({1,1,1})
  local colors = { red, green, blue, white }
  local lsm = nn.LogSoftMax():cuda()
  
  local test_images = list_files(testset_path, nil, true)
  
  -- optionally add random images from training set
  --local test_images = {}
  --[[for n=1,10 do
    local fn = image_file_names[torch.random() % #image_file_names + 1]
    table.insert(test_images, fn)
  end]]--
  
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local amp = nn.SpatialAdaptiveMaxPooling(kw, kh):cuda()

  for n,fn in ipairs(test_images) do
    
    -- load image
    local input = load_image_auto_size(fn, cfg.target_smaller_side, cfg.max_pixel_size, cfg.color_space)
    local input_size = input:size()
    local input_rect = Rect.new(0, 0, input_size[3], input_size[2])
    
    input[1] = normalization:forward(input[{{1}}])
    input = input:cuda()

    -- pass image through network
    pnet:evaluate()
    local outputs = pnet:forward(input)

    -- analyse network output for non-background classification
    local matches = {}

    local aspect_ratios = 3
    for i=1,4 do
      local layer = outputs[i]
      local layer_size = layer:size()
      for y=1,layer_size[2] do
        for x=1,layer_size[3] do
          local c = layer[{{}, y, x}]
          for a=1,aspect_ratios do

            local ofs = (a-1) * 6
            local cls_out = c[{{ofs + 1, ofs + 2}}] 
            local reg_out = c[{{ofs + 3, ofs + 6}}]
            
            -- regression
            local a = anchors:get(i,a,y,x)
            local r = Anchors.anchorToInput(a, reg_out)
            
            -- classification
            local c = lsm:forward(cls_out)
            if math.exp(c[1]) > 0.9 and r:overlaps(input_rect) then
              table.insert(matches, { p=c[1], a=a, r=r, l=i })
            end
            
          end
        end
      end      
    end
    
    local winners = {}
    
    if #matches > 0 then
    
      -- NON-MAXIMUM SUPPRESSION
      local bb = torch.Tensor(#matches, 4)
      local score = torch.Tensor(#matches, 1)
      for i=1,#matches do
        bb[i] = matches[i].r:totensor()
        score[i] = matches[i].p
      end
      
      local iou_threshold = 0.5
      local pick = nms(bb, iou_threshold, 'area')
      --local pick = nms(bb, iou_threshold, score)
      local candidates = {}
      pick:apply(function (x) table.insert(candidates, matches[x]) end )
  
      print(string.format('candidates: %d', #candidates))
      
      -- REGION CLASSIFICATION 
  
      cnet:evaluate()
      
      -- create cnet input batch
      local cinput = torch.CudaTensor(#candidates, cfg.roi_pooling.kw * cfg.roi_pooling.kh * 300)
      for i,v in ipairs(candidates) do
        -- pass through adaptive max pooling operation
        local pi, idx = extract_roi_pooling_input(v.r, localizer, outputs[5])
        cinput[i] = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * 300)
      end
      
      -- send extracted roi-data through classification network
      local coutputs = cnet:forward(cinput)
      local bbox_out = coutputs[1]
      local cls_out = coutputs[2]
      
      local yclass = {}
      for i,x in ipairs(candidates) do
        x.r2 = Anchors.anchorToInput(x.r, bbox_out[i])
        
        local cprob = cls_out[i]
        local p,c = torch.sort(cprob, 1, true) -- get probabilities and class indicies
        
        x.class = c[1]
        x.confidence = p[1]
        
        if x.class ~= bgclass then --and math.exp(x.confidence) > 0.01 then
          if not yclass[x.class] then
            yclass[x.class] = {}
          end
          
          table.insert(yclass[x.class], x)
        end
      end

      -- run per class NMS
      for i,c in pairs(yclass) do
        -- fill rect tensor
        bb = torch.Tensor(#c, 5)
        for j,r in ipairs(c) do
          bb[{j, {1,4}}] = r.r2:totensor()
          bb[{j, 5}] = r.confidence
        end
        
        pick = nms(bb, 0.1, bb[{{}, 5}])
        pick:apply(function (x) table.insert(winners, c[x]) end ) 
       
      end
      
    end

    -- load image back to rgb-space before drawing rectangles
    local img = load_image_auto_size(fn, cfg.target_smaller_side, cfg.max_pixel_size, 'rgb')
    
    for i,m in ipairs(winners) do
        --draw_rectangle(img, m.r, blue)
        draw_rectangle(img, m.r2, green)
    end
    
    image.saveJPG(string.format('dummy%d.jpg', n), img)
    
  end
end

function graph_training(cfg, snapshot_prefix, training_data_filename, network_filename)
  print('Reading training data file \'' .. training_data_filename .. '\'.')
  local training_data = load_obj(training_data_filename)
  local file_names = keys(training_data.ground_truth)
  print(string.format("Training data loaded. Dataset: '%s'; Total files: %d; classes: %d; Background: %d)", 
      training_data.dataset_name, 
      #file_names,
      #training_data.class_names,
      #training_data.background_files))
  
--[[   
  local cfg = {
    class_count = #training_data.class_names,  -- excluding background class
    target_smaller_side = 450,
    scales = { 48, 96, 192, 384 },
    --scales = { 32, 64, 128, 256 },
    max_pixel_size = 1000,
    normalization = { method = 'none' },
    color_space = 'yuv',
    roi_pooling = { kw = 6, kh = 6 },
    examples_base_path = '/data/imagenet/ILSVRC2015/Data/DET/train',
    --background_base_path = '',
    batch_size = 256,
    positive_threshold = 0.5, 
    negative_threshold = 0.25,
    best_match = true,
    nearby_aversion = true
  }]]
  
  training_data.cfg = cfg  -- add cfg
  
  local training_stats = {}
  
  local stored
  if network_filename then
    stored = load_obj(network_filename)
    --opt = stored.options
    training_stats = stored.stats
  end
  
  local pnet = create_proposal_net()
  local cnet = create_classifaction_net(cfg.roi_pooling.kw, cfg.roi_pooling.kh, 300, cfg.class_count + 1)

  pnet:cuda()
  cnet:cuda()
  
  -- combine parameters from pnet and cnet into flat tensors
  local weights, gradient = combine_and_flatten_parameters(pnet, cnet)
  if stored then
    weights:copy(stored.weights)
  end
  
  local batchIterator = BatchIterator.new(pnet, training_data)
  local eval_objective_grad = create_objective(pnet, cnet, weights, gradient, batchIterator)
  
  local rmsprop_state = { learningRate = opt.lr, alpha = opt.rms_decay }
  --local nag_state = { learningRate = opt.lr, weightDecay = 0, momentum = opt.rms_decay }
  --local sgd_state = { learningRate = 0.000025, weightDecay = 1e-7, momentum = 0.9 }
  
  for i=1,50000 do
    if i % 5000 == 0 then
      opt.lr = opt.lr / 2
      rmsprop_state.lr = opt.lr
    end
  
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(eval_objective_grad, weights, rmsprop_state)
    --local _, loss = optim.nag(optimization_target, weights, nag_state)
    --local _, loss = optim.sgd(optimization_target, weights, sgd_state)
    
    local time = timer:time().real

    table.insert(training_stats, { loss = loss[1], time = time })
    print(string.format('%d: loss: %f', i, loss[1]))
    
    if i%1000 == 0 then
      -- save snapshot
      -- todo: change weight storage (for pnet and cnet)
      save_model(string.format('%s_%06d.t7', snapshot_prefix, i), weights, opt, training_stats)
    end
    
  end
  
  -- compute positive anchors, add anchors to ground-truth file
end

--graph_training(duplo_cfg, 'duplo', 'duplo.t7', 'av_036000.t7')
graph_training(imgnet_cfg, 'imgnet', 'ILSVRC2015_DET.t7')
 
--graph_evaluate(duplo_cfg, 'duplo.t7', 'duplo_036000.t7', true, 17)
