require 'torch'
require 'pl'
require 'optim'
require 'image'
require 'nngraph'
require 'cunn'
require 'SmoothL1Criterion'
require 'nms'

require 'utilities'
require 'model'
require 'Anchors'
require 'BatchIterator'
require 'Objective'

-- parameters

local base_path = '/home/team/datasets/brickset_all/'
local testset_path = '/home/team/datasets/realbricks/'
local class_count = 16 + 1
local kw, kh = 7,7

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

function read_csv_file(fn)
  -- format of RoI file:
  -- filename, left, top, right, bottom, model_class_name, model_class_index, material_name, material_index
  -- "img8494058054b911e5a5ab086266c6c775.png", 0, 573, 59, 701, "DuploBrick_2x2", 2, "DuploBrightGreen", 11

  local f = io.open(fn, 'r')

  local filemap = {}
  
  for l in f:lines() do
    local v = l:split(',') -- get values of single row (we have a trivial csv file without ',' in string values)
    
    local image_file_name = remove_quotes(v[1])  -- extract image file name, remove quotes
    local roi_entry = {
      rect = Rect.new(tonumber(v[2]), tonumber(v[3]), tonumber(v[4]), tonumber(v[5])),
      model_class_name = remove_quotes(v[6]), 
      model_class_index = tonumber(v[7]),
      material_name = remove_quotes(v[8]),
      material_index = tonumber(v[9])
    }
    
    local file_entry = filemap[image_file_name]
    if file_entry == nil then
      file_entry = { image_file_name = image_file_name, rois = {} }
      filemap[image_file_name] = file_entry 
    end
    
    table.insert(file_entry.rois, roi_entry)
  end
 
  f:close()
  
  return filemap
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


function precompute_positive_list(out_fn, positive_threshold, negative_threshold, test_size)
  local target_smaller_side = 450
  local max_pixel_size = 1000
  local scales = { 48, 96, 192, 384 }
   
  local roi_file_name = path.join(base_path, 'boxes.csv') 
  local ground_truth = read_csv_file(roi_file_name)
  local image_file_names = keys(ground_truth)
  
  -- determine layer sizes
  local pnet = create_proposal_net()
  local anchors = Anchors.new(pnet, scales)
    
  for n,x in pairs(ground_truth) do
    local img, original_size = load_image_auto_size(n, target_smaller_side, max_pixel_size, 'rgb')
    x.scaleX = img:size()[3] / original_size[3]
    x.scaleY = img:size()[2] / original_size[2]
    local rois = x.rois
    for i=1,#rois do
      rois[i].original_rect = rois[i].rect
      rois[i].rect = rois[i].rect:scale(x.scaleX, x.scaleY)
    end
    x.positive_anchors = anchors:findPositive(rois, Rect.new(0, 0, img:size()[3], img:size()[2]), positive_threshold, negative_threshold, true)
    print(string.format('%s: %d (rois: %d)', n, #x.positive_anchors, #x.rois))
  end
  
  test_size = test_size or 0.2 -- 80:20 split
  if test_size >= 0 and test_size < 1 then
    test_size = math.ceil(#image_file_names * test_size)
  end
  shuffle(image_file_names)
  local test_set = remove_tail(image_file_names, test_size)
  
  local training_data = 
  {
    target_smaller_side = target_smaller_side, 
    max_pixel_size = max_pixel_size,
    scales = scales,
    train_file_names = image_file_names,
    test_file_name = test_set,
    ground_truth = ground_truth
  }
  save_obj(out_fn, training_data)
end

function graph_evaluate(training_data_filename, network_filename, normalize, bgclass)
  local training_data = load_obj(training_data_filename)
  local ground_truth = training_data.ground_truth
  local image_file_names = training_data.image_file_names

  local stored = load_obj(network_filename)
  
  local class_count = class_count
  local pnet = create_proposal_net()
  local cnet = create_classifaction_net(kw, kh, 300, class_count)
  pnet:cuda()
  cnet:cuda()
  
  local anchors = Anchors.new(pnet, training_data.scales)
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
  
  local amp = nn.SpatialAdaptiveMaxPooling(7, 7):cuda()
  for n,fn in ipairs(test_images) do
    
    -- load image
    local input = load_image_auto_size(fn, training_data.target_smaller_side, training_data.max_pixel_size, 'yuv')
    local input_size = input:size()
    local input_rect = Rect.new(0, 0, input_size[3], input_size[2])
    input = normalize_image(input):cuda()

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
            if math.exp(c[1]) > 0.95 and r:overlaps(input_rect) then
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
      for i=1,#matches do
        bb[i] = matches[i].r:totensor()
      end
      
      local iou_threshold = 0.5
      local pick = nms(bb, iou_threshold, 'area')
      local candidates = {}
      pick:apply(function (x) table.insert(candidates, matches[x]) end )
  
      -- REGION CLASSIFICATION 
  
      cnet:evaluate()
      
      -- create cnet input batch
      local cinput = torch.CudaTensor(#candidates, 7 * 7 * 300)
      for i,v in ipairs(candidates) do
        -- pass through adaptive max pooling operation
        local pi, idx = extract_roi_pooling_input(v.r, localizer, outputs[5])
        cinput[i] = amp:forward(pi):view(7 * 7 * 300)
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
        
        if x.class ~= bgclass and math.exp(x.confidence) > 0.2 then
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
    local img = load_image_auto_size(fn, training_data.target_smaller_side, training_data.max_pixel_size, 'rgb')
    
    for i,m in ipairs(winners) do
        --draw_rectangle(img, m.r, blue)
        draw_rectangle(img, m.r2, green)
    end
    
    image.saveJPG(string.format('dummy%d.jpg', n), img)
    
  end
end

function graph_training(network_filename)
  
  -- TODO: create function to create training configuration
  local ground_truth = read_csv_file('/home/team/datasets/brickset_all/boxes.csv')
  local image_file_names = keys(ground_truth)
  
  test_size = test_size or 0.2 -- 80:20 split
  if test_size >= 0 and test_size < 1 then
    test_size = math.ceil(#image_file_names * test_size)
  end
  shuffle(image_file_names)
  local test_set = remove_tail(image_file_names, test_size)
  
  local background_file_names = list_files('/home/team/datasets/background', nil, true)
  
  local cfg = {
    class_count = 16,  -- excluding background class
    target_smaller_side = 450,
    --scales = { 48, 96, 192, 384 },
    scales = { 32, 64, 128, 256 },
    max_pixel_size = 1000,
    normalization = { method = 'contrastive', width = 7 },
    color_space = 'yuv',
    roi_pooling = { kw = 6, kh = 6 },
    examples_base_path = '/home/team/datasets/brickset_all/',
    background_base_path = '/home/team/datasets/background/',
    batch_size = 256,
    positive_threshold = 0.5, 
    negative_threshold = 0.25,
    best_match = true,
    nearby_aversion = true
  }
  
  local training_data = {
    ground_truth = ground_truth, 
    train_file_names = image_file_names,
    test_file_name = test_set,
    background_file_names = background_file_names,
    cfg = cfg
  }
  
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
      save_model(string.format('full2_%06d.t7', i), weights, opt, training_stats)
    end
    
  end
  
  -- compute positive anchors, add anchors to ground-truth file
end

--precompute_positive_list('training_data.t7', 0.6, 0.3)
graph_training() 
--graph_evaluate('training_data.t7', 'full2_026000.t7', true, 17)
