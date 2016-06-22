require 'torch'
require 'optim'
require 'image'
require 'cunn'

require 'utilities'
require 'Anchors'
require 'models.model_utilities'

cmd = torch.CmdLine()
cmd:addTime()

cmd:text()
cmd:text('Training a convnet for region proposals')
cmd:text()

cmd:text('=== Training ===')
cmd:option('-cfg', 'config/imagenet.lua', 'configuration file')
cmd:option('-model', 'models/vgg_small.lua', 'model factory file')
cmd:option('-name', 'imgnet', 'experiment name, snapshot prefix')
cmd:option('-train', 'receptive_fields_data.t7', 'training data file name')
cmd:option('-resultDir', 'proposal_logs', 'Folder for storing all result. (training process ect)')
cmd:option('-lr', 1E-3, 'learn rate')

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

-- create result directory
os.execute(('mkdir -p %s'):format(opt.resultDir))

-- system configuration
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpuid + 1)  -- nvidia tools start counting at 0
torch.setnumthreads(opt.threads)
if opt.seed ~= nil and opt.seed ~= 0 then
  torch.manualSeed(opt.seed)
  cutorch.manualSeed(opt.seed)
end


function normalizeImageInplace(img)
  for i = 1,img:size(1) do
    img[i]:add(-img[i]:mean())
    local s = img[i]:std()
    if s > 1e-8 then
      img[i]:div(s)
    end
  end
end


function prepareTrainingBatch(training_data, offsets, start_class, end_class, w, h)
  local nExamples = end_class-start_class+1

  local X = torch.FloatTensor(nExamples, 3, h, w)
  local Y = torch.LongTensor(nExamples)

  -- sample from classes
  local j = 1
  for i=start_class,end_class do
    offsets[i] = offsets[i] + 1
    if offsets[i] > #training_data.training_by_class[i] then
      offsets[i] = 1
    end

    local fn = training_data.training_by_class[i][offsets[i]]

    local img = load_image(fn, 'yuv')
    normalizeImageInplace(img)
    X[j] = img
    Y[j] = i
    j = j + 1
  end

  return X, Y
end


function prepareValidationBatch(training_data, nExamples, w, h)
  local X = torch.FloatTensor(nExamples, 3, h, w)
  local Y = torch.LongTensor(nExamples)

  for i=1,nExamples do
    -- sample from validaten set
    local j = torch.random(#training_data.validation_set)
    local fn = training_data.validation_set[j]
    local img = load_image(fn, 'yuv')
    normalizeImageInplace(img)
    X[i] = img
    Y[i] = training_data.ground_truth[fn].class_index
  end

  return X, Y
end


function simpleProposalPretraining(cfg, model_path, snapshot_prefix, training_data_filename, network_filename)
  local w,h = 122,122

  -- load training data
  local trainig_data = torch.load(training_data_filename)

  local class_count = #trainig_data.training_by_class

  -- create model, expected 8x8 output plane size currently hard coded
  local _1, layers, _2, _3 = dofile(model_path)
  local net = create_simple_pretraining_net(layers, 8 * 8 * layers[#layers].filters, class_count)
  
  -- create criterion
  local criterion = nn.ClassNLLCriterion()

  -- convert model to CUDA
  criterion:cuda()
  net:cuda()

  local weights, gradient = net:getParameters()

  -- clean up before training
  collectgarbage()

  -- prepare optimization method

  local optimState = {
    learningRate = opt.lr,
    weightDecay = 1E-5,
    momentum = 0.6,
    learningRateDecay = 0
  }
  optimMethod = optim.sgd

--[[  local optimState = {
    learningRate = opt.lr,
    beta1 = 0.9,      -- first moment coefficient
    beta2 = 0.999     -- second moment coefficient
  }
  local optimMethod = optim.adam]]

  local function randomizeOffsets()
    local offsets = {}
    for i=1,class_count do
      table.insert(offsets, torch.random(#trainig_data.training_by_class[i]))
    end
    return offsets
  end

  -- generate inital offsets
  local offsets = randomizeOffsets()

  local BATCH_SIZE = 75

  local function lossAndGradient(x)
    if x ~= weights then
      weights:copy(x)
    end
    net:zeroGradParameters()
    net:training()

    local i = 0
    local loss = 0
    BATCH_SIZE = 5
    start_class=1
    for start_class=1,#offsets,BATCH_SIZE do
      -- prepare batch
      local X,Y = prepareTrainingBatch(trainig_data, offsets, start_class, math.min(start_class + BATCH_SIZE-1, #offsets), w, h)
      X,Y = X:cuda(),Y:cuda()

      -- forward & backward pass
      local net_output = net:forward(X)
      loss = loss + criterion:forward(net_output, Y)
      net:backward(X, criterion:backward(net_output, Y))

      i = i + 1
    end
    gradient:div(i)
    return loss / i, gradient
  end

  -- main training loop
  for i=1,1000 do
    if i%500 == 0 then
      randomizeOffsets()
    end

    net:evaluate()

    local timer = torch.Timer()
    local _, loss = optimMethod(lossAndGradient, weights, optimState)
    local time = timer:time().real
    print(string.format('[Training] %d: loss: %f', i, loss[1]))

    -- check test interval
    if i%100 == 0 then
      local X,Y = prepareValidationBatch(trainig_data, 75, w, h)
      X,Y = X:cuda(),Y:cuda()
      local test_loss = criterion:forward(net:forward(X), Y)
      print(string.format('[Test] %d: loss: %f', i, test_loss))
    end

  end

end


simpleProposalPretraining(cfg, opt.model, 'pretrain_' .. opt.name, opt.train, opt.restore)


function load_model(cfg, model_path, network_filename, cuda)
  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg)

  if cuda then
    model.pnet:cuda()
  end

  local weights, gradient = model.pnet:getParameters()

  local training_stats

  -- load existing weights
  if network_filename ~= nil and #network_filename > 0 then
    local stored = load_obj(network_filename)
    training_stats = stored.stats
    weights:copy(stored.weights)
  end

  return model, weights, gradient, training_stats
end

--[[
function train_receptive_fields(cfg, model_path, snapshot_prefix, training_data_filename, network_filename)
  -- load training data
  local trainig_file = torch.load(training_data_filename)

  -- create network
  local model, weights, gradient, training_stats = load_model(cfg, model_path, network_filename, true)
  if not training_stats then
    training_stats = { pcls={}, preg={} }
  end
end


train_receptive_fields(cfg, opt.model, opt.name, opt.train, opt.restore)
]]


-- generate receptive fields
local function extract_receptive_fields(cfg, model_path, training_data_filename)
  local training_data = load_obj(training_data_filename)

  local model, weights, gradient, training_stats = load_model(cfg, model_path, nil, true)

  -- find anchor size
  local anchors = Anchors.new(model.pnet, cfg.scales)
  local localizer = anchors.localizers[1]

  local receptive_field_size = localizer:featureToInputRect(0,0,1,1)
  local w,h = receptive_field_size:size()

  -- create output directories with class id
  local class_dir_names = {}
  for i=1,cfg.class_count do
    local dir_name = string.format("%04d", i)
    lfs.mkdir(dir_name)
    table.insert(class_dir_names, dir_name)
  end

  -- TODO: build and store new ground-truth data
  local data = {
    ground_truth = {},
    training_set = {},
    training_by_class = {},
    validation_set = {},
    background_files = training_data.background_files
  }

  local j = 0

  local ts = {}
  for i,fn in ipairs(training_data.training_set) do
    ts[fn] = true
  end

  -- load image one by one from traning set and try to crop rois with full receptive field
  for fn,file_entry in pairs(training_data.ground_truth) do
    local rois = file_entry.rois
    local is_training_entry = (ts[fn] ~= nil)

    -- load image
    local status, img = pcall(function () return load_image(fn, 'rgb', cfg.examples_base_path) end)
    if status then

      local img_size = img:size()
      local image_rect = Rect.new(0, 0, img_size[3], img_size[2])

      -- extract rois with receptive field
      for i,roi in ipairs(rois) do
        local cx,cy = roi.rect:center()
        local crop_rect = Rect.fromCenterWidthHeight(cx, cy, w, h)

        if image_rect:contains(crop_rect) then

          -- extract receptive field at roi center position
          local part = img[{{}, {crop_rect.minY+1, crop_rect.maxY}, {crop_rect.minX+1, crop_rect.maxX}}]
          local fn_ = path.splitext(path.basename(fn))

          local part_fn = string.format('%s/%s_%d.jpg', class_dir_names[roi.class_index], fn_, i)
          print(part_fn)
          image.saveJPG(part_fn, part)

          data.ground_truth[part_fn] = {
            class_index = roi.class_index,
            rect = Rect.fromCenterWidthHeight(w/2, h/2, roi.rect:width(), roi.rect:height()),
            training = is_training_entry
          }

          print(data.ground_truth[part_fn])

          if is_training_entry == true then
            table.insert(data.training_set, part_fn)
            if data.training_by_class[roi.class_index] == nil then
              data.training_by_class[roi.class_index] = {}
            end
            table.insert(data.training_by_class[roi.class_index], part_fn)
          else
            table.insert(data.validation_set, part_fn)
          end

        end
      end

    end

    j = j + 1
    if (j%50000) == 0 then
      torch.save('receptive_fields_data.t7', data)
    end
  end

  torch.save('receptive_fields_data.t7', data)

end

--extract_receptive_fields(cfg, opt.model, 'ILSVRC2015_DET.t7')
