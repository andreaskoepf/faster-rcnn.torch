require 'torch'
require 'optim'
require 'image'
require 'cunn'
require 'gnuplot'

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
cmd:option('-opti', 'sgd', 'Optimizer')
cmd:option('-resultDir', 'proposal_logs', 'Folder for storing all result. (training process ect)')

cmd:text('=== Misc ===')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-gpuid', 1, 'device ID (CUDA), (use -1 for CPU)')
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


function loadImage(fn, augment)
  local img = image.load(fn)

  if augment then
    if math.random() > 0.5 then
      img = image.hflip(img)
    end
  end

  for i = 1,img:size(1) do
    img[i]:add(-img[i]:mean())
    local s = img[i]:std()
    if s > 1e-8 then
      img[i]:div(s)
    end
  end

  return img
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

    local img = loadImage(fn, true)
    X[j] = img
    Y[j] = i
    j = j + 1
  end

  return X, Y
end


function prepareTrainingBatch2(training_data, count, w, h)
  -- random sampling
  local X = torch.FloatTensor(count, 3, h, w)
  local Y = torch.LongTensor(count)

  for i=1,count do
    -- sample from training set
    local fn = training_data.training_set[torch.random(#training_data.training_set)]
    local img = loadImage(fn, true)
    X[i] = img
    Y[i] = training_data.ground_truth[fn].class_index
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
    X[i] = loadImage(fn, false)
    Y[i] = training_data.ground_truth[fn].class_index
  end

  return X, Y
end


function generateReport(out_dir, step, train_confusion, valid_confusion)
  local file = io.open(string.format('%s/report.html', out_dir),'w')
  file:write(string.format([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="progress.png">
    ]], out_dir, step))

  -- write confusion matrix
  file:write'<pre>\n'
  file:write'Training Confusion Matrix\n'
  file:write(tostring(train_confusion)..'\n')
  file:write'</pre>\n'

  file:write'<pre>\n'
  file:write'Validation Confusion Matrix\n'
  file:write(tostring(valid_confusion)..'\n')
  file:write'</pre>\n'

  file:write'</body></html>'
  file:close()
end


function simpleProposalPretraining(cfg, model_path, snapshot_prefix, training_data_filename, network_filename)
  local BATCH_SIZE = 20
  local TEST_INTERVAL = 100
  local FULL_TEST_INTERVAL = 2000
  local SNAPSHOT_INTERVAL = 5000

  local w,h = 228,228

  -- load training data
  local training_data = torch.load(training_data_filename)

  local class_count = #training_data.training_by_class -- 200

  -- create model, output plane size currently hard coded
  local _1, layers, _2, _3 = dofile(model_path)
  local net = create_simple_pretraining_net(layers, 15 * 15 * layers[#layers].filters, class_count)

  -- create criterion
  local criterion = nn.ClassNLLCriterion()

  -- convert model to CUDA
  criterion:cuda()
  net:cuda()

  local weights, gradient = net:getParameters()

  -- clean up before training
  collectgarbage()

  -- prepare optimization method

  local optim_state, optim_method, learn_schedule
  if opt.opti == 'sgd' then
    optim_state = {
      learningRate = 1E-3,
      weightDecay = 1E-5,
      momentum = 0.8,
      nesterov = true,
      learningRateDecay = 0,
      dampening = 0.0
    }
    optim_method = optim.sgd
    learn_schedule =
    {
      --  start,     end,     LR,     WD
        {     1,    8000,   5e-2,   5e-4 },
        {  8001,   16000,   1e-2,   1e-4 },
        { 16001,   24000,   5e-3,   5e-5 },
        { 24001,   35000,   1e-4,   1e-5 },
        { 35001,     1e8,   1e-5,      0 }
    }
  elseif opt.opti == 'adam' then
    optim_state = {
      beta1 = 0.9,      -- first moment coefficient
      beta2 = 0.999     -- second moment coefficient
    }
    optim_method = optim.adam
    learn_schedule =
    {
      --  start,     end,     LR,     WD
        {     1,    8000,   1e-3,   5e-4 },
        {  8001,   16000,   5e-4,   1e-4 },
        { 16001,   24000,   1e-4,   5e-5 },
        { 24001,   35000,   5e-5,   1e-5 },
        { 35001,     1e8,   1e-6,      0 }
    }
  else
    error('unsupported optimization method')
  end

  local function randomizeOffsets()
    local offsets = {}
    for i=1,class_count do
      table.insert(offsets, torch.random(#training_data.training_by_class[i]))
    end
    return offsets
  end

  -- generate inital offsets
  local offsets = randomizeOffsets()
  local train_confusion = optim.ConfusionMatrix(class_count)
  local valid_confusion = optim.ConfusionMatrix(class_count)
  local stats = { train = {}, valid = {} }
  train_confusion:zero()
  valid_confusion:zero()

  local function lossAndGradient(x)
    if x ~= weights then
      weights:copy(x)
    end
    net:zeroGradParameters()
    net:training()

    local i = 0
    local loss = 0

    for start_class=1,#offsets,BATCH_SIZE do
      -- prepare batch
      local X, Y
      --if #stats.train % 2 == 0 then
        -- uniform random
        X,Y = prepareTrainingBatch2(training_data, BATCH_SIZE, w, h)
      --[[else
        -- stratified
        X,Y = prepareTrainingBatch(training_data, offsets, start_class, math.min(start_class + BATCH_SIZE-1, #offsets), w, h)
      end]]
      X,Y = X:cuda(),Y:cuda()

      -- forward & backward pass
      local net_output = net:forward(X)
      train_confusion:batchAdd(net_output, Y)
      loss = loss + criterion:forward(net_output, Y)
      net:backward(X, criterion:backward(net_output, Y))
      collectgarbage()

      i = i + 1
    end

    gradient:div(i)
    return loss / i, gradient
  end

  local function setLearnParams(learn_schedule, step, optim_state)
    for _,row in ipairs(learn_schedule) do
      if step >= row[1] and step <= row[2] then
        optim_state.learningRate = row[3]
        optim_state.weightDecay = row[4]
      end
    end
  end

  local function plotProgress(stats)
    local fn = string.format('%s/progress.png', opt.resultDir)
    gnuplot.pngfigure(fn)
    gnuplot.title('Traning progress over time')

    local xs = torch.range(1, #stats.train)

    local train = torch.Tensor(stats.train)
    local valid = torch.Tensor(stats.valid)

    gnuplot.plot(
      { 'train', train[{{},1}], train[{{},2}], '-' },
      { 'valid', valid[{{},1}], valid[{{},2}], '-' }
    )

    --gnuplot.axis({ 0, train[{stats.train,1}], 0, 10 })
    gnuplot.xlabel('iteration')
    gnuplot.ylabel('loss')

    gnuplot.plotflush()
  end

  -- main training loop
  for i=1,50000 do
    if i%500 == 0 then
      randomizeOffsets()
    end

    setLearnParams(learn_schedule, i, optim_state)

    local timer = torch.Timer()
    local _, loss = optim_method(lossAndGradient, weights, optim_state)
    local time = timer:time().real
    table.insert(stats.train, { i, loss[1] })
    print(string.format('[Training] %d: loss: %f', i, loss[1]))

    -- check test interval
    if i%FULL_TEST_INTERVAL == 0 then
      valid_confusion:zero()

      net:evaluate()

      -- use all validation images
      local BATCH_SIZE = 20
      local X = torch.FloatTensor(BATCH_SIZE, 3, h, w)
      local Y = torch.LongTensor(BATCH_SIZE)
      for i=1,#training_data.validation_set,BATCH_SIZE do
        for j=1,BATCH_SIZE do
          if i+j > #training_data.validation_set then break end
          local fn = training_data.validation_set[i+j]
          X[j] = loadImage(fn, false)
          Y[j] = training_data.ground_truth[fn].class_index
        end

        local X_,Y_ = X:cuda(),Y:cuda()
        local net_output = net:forward(X_)
        valid_confusion:batchAdd(net_output, Y_)
      end

      valid_confusion:updateValids()
    end

    if i%TEST_INTERVAL == 0 then
      local valid_loss = 0
      local TEST_BATCH_COUNT = 50
      net:evaluate()
      for j=1,TEST_BATCH_COUNT do
        local X,Y = prepareValidationBatch(training_data, 20, w, h)
        X,Y = X:cuda(),Y:cuda()
        local net_output = net:forward(X)
        valid_loss = valid_loss + criterion:forward(net_output, Y)
      end
      valid_loss = valid_loss / TEST_BATCH_COUNT
      table.insert(stats.valid, { i, valid_loss })

      plotProgress(stats)
      train_confusion:updateValids()
      generateReport(opt.resultDir, i, train_confusion, valid_confusion)

      print(string.format('[Test] %d: loss: %f', i, valid_loss))
      train_confusion:zero()
    end

    if i%SNAPSHOT_INTERVAL == 0 then
      local snapshot_fn = string.format('%s/snapshot_%07d.t7', opt.resultDir, i)
      torch.save(snapshot_fn, {
        version = 0,
        weights = weights,
        options = opt,
        stats = stats
      })
    end

  end

end


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
  local training_file = torch.load(training_data_filename)

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

  -- debug graph structure
  -- graph.dot(model.pnet.fg, 'test', 'graph_out')

  -- find anchor size
  local anchors = Anchors.new(model.pnet, cfg.scales)
  local localizer = anchors.localizers[1]

  local receptive_field_size = localizer:featureToInputRect(0,0,1,1)
  local w,h = receptive_field_size:size()
  print('receptive field size: ' .. tostring(receptive_field_size))

  -- create output directories with class id
  local class_dir_names = {}
  for i=1,cfg.class_count do
    local dir_name = string.format("%04d", i)
    lfs.mkdir(dir_name)
    table.insert(class_dir_names, dir_name)
  end

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


-- step 1: extract receptive fields
--extract_receptive_fields(cfg, opt.model, 'ILSVRC2015_DET.t7')

-- step 2: pretraining
simpleProposalPretraining(cfg, opt.model, 'pretrain_' .. opt.name, opt.train, opt.restore)

