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

-- parameters

local base_path = '/home/koepf/datasets/brickset_all/'
local testset_path = '/home/koepf/datasets/realbricks/'
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

local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(11))

function normalize_image(img)
  img[1] = normalization:forward(img[{{1}}])
  return img
end

function load_image(fn, w, h, normalize)
  local img = image.load(path.join(base_path, fn), 3, 'float')
  local originalSize = img:size()
  img = image.rgb2yuv(image.scale(img, w, h))
  if normalize then
    img[1] = normalization:forward(img[{{1}}])
  end
  local scaleX, scaleY = w / originalSize[3], h / originalSize[2]
  return img, scaleX, scaleY
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

--[[function compute_feature_layer_rect(input_rect, input_size, feature_layer_size)
  local input_height, input_width = input_size[2], input_size[3]
  local h, w = feature_layer_size[2], feature_layer_size[3]
  local scaleY, scaleX = h / input_height, w / input_width
  return Rect.new(input_rect):scale(scaleX, scaleY):snapToInt():clip(Rect.new(0, 0, w, h))
end]]

function extract_roi_pooling_input(input_rect, localizer, feature_layer_output)
  local r = localizer:inputToFeatureRect(input_rect)
  -- the use of math.min ensures correct handling of empty rects, 
  -- +1 offset for top, left only is conversion from half-open 0-based interval
  local s = feature_layer_output:size()
  r = r:clip(Rect.new(0,0,s[3],s[2]))
  local idx = { {}, { math.min(r.minY + 1, r.maxY), r.maxY }, { math.min(r.minX + 1, r.maxX), r.maxX } }
  return feature_layer_output[idx], idx
end

function create_optimization_target(pnet, cnet, weights, gradient, training_data, normalize, bgclass)
  local ground_truth = training_data.ground_truth 
  local train_file_names =  training_data.train_file_names
  local test_file_names =  training_data.test_file_names
  
  local anchors = Anchors.new(pnet, training_data.scales)
  local localizer = Localizer.new(pnet.outnode.children[5])
  
  local softmax = nn.CrossEntropyCriterion():cuda()
  local cnll = nn.ClassNLLCriterion():cuda()
  local smoothL1 = nn.SmoothL1Criterion():cuda()
  smoothL1.sizeAverage = false
  
  local amp = nn.SpatialAdaptiveMaxPooling(kw, kh):cuda()

  local function loss_and_gradient(w)
    if w ~= weights then
      weights:copy(w)
    end
    gradient:zero()
    
    local cls_loss = 0
    local reg_loss = 0 
    local delta_outputs = { }
    local cls_count = 0
    local reg_count = 0
    
    local creg_loss, creg_count = 0, 0
    local ccls_loss, ccls_count = 0, 0
    
    pnet:training()
    cnet:training()
    
    while cls_count < 256 do
    
      -- select random image
      local fn = train_file_names[torch.random() % #train_file_names + 1]
      local rois = ground_truth[fn].rois
     
      -- load image
      --print(fn)
      local img = load_image_auto_size(fn, training_data.target_smaller_side, training_data.max_pixel_size, 'yuv')
      local img_size = img:size()
      img = normalize_image(img)
      
      -- get positive and negative anchors examples
      local p = ground_truth[fn].positive_anchors
      local n = anchors:sampleNegative(Rect.new(0, 0, img_size[3], img_size[2]), rois, 0.3, math.max(16, #p))
      
      -- convert batch to cuda if we are running on the gpu
      img = img:cuda()
      
      -- run forward convolution
      local outputs = pnet:forward(img)
      
      -- clear delta values
      for i,out in ipairs(outputs) do
        if not delta_outputs[i] then
          delta_outputs[i] = torch.FloatTensor():cuda()
        end
        delta_outputs[i]:resizeAs(out)
        delta_outputs[i]:zero()
      end
     
     local roi_pool_state = {}
     local input_size = img:size()
     local cnetgrad
     
      -- process positive set
      for i,x in ipairs(p) do
        local anchor = x[1]
        local roi = x[2]
        local l = x[1].layer
        
        local out = outputs[l]
        local delta_out = delta_outputs[l]
         
        local idx = x[1].index
        local v = out[idx]
        local d = delta_out[idx]
          
        -- classification
        cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 1)
        local dc = softmax:backward(v[{{1, 2}}], 1)
        d[{{1,2}}]:add(dc)
        
        -- box regression
        local reg_out = v[{{3, 6}}]
        local reg_target = Anchors.inputToAnchor(anchor, roi.rect):cuda()  -- regression target
        local reg_proposal = Anchors.anchorToInput(anchor, reg_out)
        reg_loss = reg_loss + smoothL1:forward(reg_out, reg_target) * 10
        local dr = smoothL1:backward(reg_out, reg_target) * 10
        d[{{3,6}}]:add(dr)
        
        -- pass through adaptive max pooling operation
        local pi, idx = extract_roi_pooling_input(roi.rect, localizer, outputs[5])
        local po = amp:forward(pi):view(7 * 7 * 300)
        table.insert(roi_pool_state, { input = pi, input_idx = idx, anchor = anchor, reg_proposal = reg_proposal, roi = roi, output = po:clone(), indices = amp.indices:clone() })
      end
      
      -- process negative
      for i,x in ipairs(n) do
        local l = x.layer
        local out = outputs[l]
        local delta_out = delta_outputs[l]
        local idx = x.index
        local v = out[idx]
        local d = delta_out[idx]
        
        cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 2)
        local dc = softmax:backward(v[{{1, 2}}], 2)
        d[{{1,2}}]:add(dc)
        
        -- pass through adaptive max pooling operation
        local pi, idx = extract_roi_pooling_input(x, localizer, outputs[5])
        local po = amp:forward(pi):view(7 * 7 * 300)
        table.insert(roi_pool_state, { input = pi, input_idx = idx, output = po:clone(), indices = amp.indices:clone() })
      end
      
      -- send extracted roi-data through classification network
      
      -- create cnet input batch
      local cinput = torch.CudaTensor(#roi_pool_state, kh * kw * 300)
      local cctarget = torch.CudaTensor(#roi_pool_state)
      local crtarget = torch.CudaTensor(#roi_pool_state, 4):zero()
      
      for i,x in ipairs(roi_pool_state) do
        cinput[i] = x.output
        if x.roi then
          -- positive example
          cctarget[i] = x.roi.model_class_index + 1
          crtarget[i] = Anchors.inputToAnchor(x.reg_proposal, x.roi.rect)   -- base fine tuning on proposal
        else
          -- negative example
          cctarget[i] = bgclass
        end
      end
      
      -- process classification batch 
      local coutputs = cnet:forward(cinput)
      
      -- compute classification and regression error and run backward pass
      local crout = coutputs[1]
      --print(crout)
      
      crout[{{#p + 1, #roi_pool_state}, {}}]:zero() -- ignore negative examples
      creg_loss = creg_loss + smoothL1:forward(crout, crtarget) * 10
      local crdelta = smoothL1:backward(crout, crtarget) * 10
      
      local ccout = coutputs[2]  -- log softmax classification
      local loss = cnll:forward(ccout, cctarget)
      ccls_loss = ccls_loss + loss 
      local ccdelta = cnll:backward(ccout, cctarget)
      
      local post_roi_delta = cnet:backward(cinput, { crdelta, ccdelta })
      
      -- run backward pass over rois
      for i,x in ipairs(roi_pool_state) do
        amp.indices = x.indices
        delta_outputs[5][x.input_idx]:add(amp:backward(x.input, post_roi_delta[i]:view(300, kh, kw)))
      end
      
      -- backward pass of proposal network
      local gi = pnet:backward(img, delta_outputs)
      -- print(string.format('%f; pos: %d; neg: %d', gradient:max(), #p, #n))
      reg_count = reg_count + #p
      cls_count = cls_count + #p + #n
      
      creg_count = creg_count + #p
      ccls_count = ccls_count + 1
    end
   
    -- scale gradient
    gradient:div(cls_count)
    
    print(string.format('prop: cls: %f (%d), reg: %f (%d); det: cls: %f, reg: %f', 
      cls_loss / cls_count, cls_count, reg_loss / reg_count, reg_count,
      ccls_loss / ccls_count, creg_loss / creg_count)
    )
    
    local loss = cls_loss / cls_count + reg_loss / reg_count
    return loss, gradient
  end
  
  return loss_and_gradient
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

function graph_evaluate(training_data_filename, network_filename, normalize)
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
  
  local test_images = list_files(testset_path)
  
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
            if math.exp(c[1]) > 0.9 then
              table.insert(matches, { p=c[1], a=a,  r=r, l=i })
            end
            
          end
        end
      end      
    end
    
    local winners = {}
    
    print(#matches)
    if #matches > 0 then
    
      -- NON-MAXIMUM SUPPRESSION
      local bb = torch.Tensor(#matches, 4)
      for i=1,#matches do
        bb[i] = matches[i].r:totensor()
      end
      
      local iou_threshold = 0.5
      local pick = nms(bb, iou_threshold, 'area')
      
      pick:apply(function (x) table.insert(winners, matches[x]) end )
  
      -- REGION CLASSIFICATION 
  
      cnet:evaluate()
      
      -- create cnet input batch
      local cinput = torch.CudaTensor(#winners, 7 * 7 * 300)
      for i,v in ipairs(winners) do
        -- pass through adaptive max pooling operation
        local pi, idx = extract_roi_pooling_input(v.r, localizer, outputs[5])
        local po = amp:forward(pi):view(7 * 7 * 300)
        cinput[i] = po:clone()
      end
      
      -- send extracted roi-data through classification network
      local coutputs = cnet:forward(cinput)
      
      -- compute classification and regression error and run backward pass
      local bbox_out = coutputs[1]
      local cls_out = coutputs[2]
      
      for i=1,#winners do
        winners[i].r2 = Anchors.anchorToInput(winners[i].a, bbox_out[i])
        
        local cprob = cls_out[i]
        local p,c = torch.sort(cprob, 1, true) -- get probabilities and class indicies
        
        winners[i].class = c[1]
        winners[i].confidence = p[1]
      end

    end

    -- load image back to rgb-space before drawing rectangles
    local img = load_image_auto_size(fn, training_data.target_smaller_side, training_data.max_pixel_size, 'rgb')
    
    for i,m in ipairs(winners) do
      local color
      if m.class ~= 17 and math.exp(m.confidence) > 0.25 then
        draw_rectangle(img, m.r, blue)
        --draw_rectangle(img, m.r2, green)
      end
    end
    
    image.saveJPG(string.format('dummy%d.jpg', n), img)
    
  end
end

function graph_training(training_data_filename, network_filename)
  local training_data = load_obj(training_data_filename)
  local training_stats = {}
  
  local stored
  if network_filename then
    stored = load_obj(network_filename)
    --opt = stored.options
    training_stats = stored.stats
  end
  
  local class_count = 16 + 1
  local pnet = create_proposal_net()
  local cnet = create_classifaction_net(kw, kh, 300, class_count)
  
  if opt.gpuid >= 0 then
    pnet:cuda()
    cnet:cuda()
  end
  
  -- combine parameters from pnet and cnet into flat tensors
  local weights, gradient = combine_and_flatten_parameters(pnet, cnet)
  
  print(weights:size())
  print(gradient:size())
  
  if stored then
    weights:copy(stored.weights)
  end
  
  local rmsprop_state = { learningRate = opt.lr, alpha = opt.rms_decay }
  --local nag_state = { learningRate = opt.lr, weightDecay = 0, momentum = opt.rms_decay }
  --local sgd_state = { learningRate = 0.000025, weightDecay = 1e-7, momentum = 0.9 }
  
  local optimization_target = create_optimization_target(pnet, cnet, weights, gradient, training_data, true, class_count)
  
  for i=1,50000 do
  
    if i % 5000 == 0 then
      opt.lr = opt.lr / 2
      rmsprop_state.lr = opt.lr
    end
  
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(optimization_target, weights, rmsprop_state)
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
graph_training('training_data.t7') 
--graph_evaluate('training_data.t7', 'full2_022000.t7', true)
