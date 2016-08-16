require 'torch'
require 'pl'
require 'lfs'
local optim = require 'optim'
local image = require 'image'
require 'nngraph'
require 'cunn'
require 'nms'
require 'gnuplot'

require 'utilities'
require 'Anchors'
require 'BatchIterator'
require 'objective'
require 'Detector'
local c = require 'trepl.colorize'

-- command line options
cmd = torch.CmdLine()
cmd:addTime()

cmd:text()
cmd:text('Training a convnet for region proposals')
cmd:text()

cmd:text('=== Training ===')
cmd:option('-cfg', 'config/imagenet.lua', 'configuration file')
cmd:option('-model', 'models/vgg_small.lua', 'model factory file')
cmd:option('-name', 'imgnet', 'experiment name, snapshot prefix')
cmd:option('-train', 'ILSVRC2015_DET.t7', 'training data file name')
cmd:option('-restore', '', 'network snapshot file name to load')
cmd:option('-mode', 'both', 'one of three training modes: onlyPnet, onlyCnet, or both')
cmd:option('-snapshot', 1000, 'snapshot interval')
cmd:option('-plot', 100, 'plot training progress interval')
cmd:option('-lr', 1E-3, 'learn rate')
cmd:option('-rms_decay', 0.9, 'RMSprop moving average dissolving factor')
cmd:option('-opti', 'adam', 'Optimizer')
cmd:option('-resultDir', 'logs', 'Folder for storing all result. (training progress etc.)')

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

local confusion_pcls = optim.ConfusionMatrix(2)-- background and forground
local confusion_ccls = optim.ConfusionMatrix(cfg.class_count+1)-- background and forground
-- system configuration
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpuid + 1)  -- nvidia tools start counting at 0
torch.setnumthreads(opt.threads)
if opt.seed ~= 0 then
  torch.manualSeed(opt.seed)
  cutorch.manualSeed(opt.seed)
end

local function averagePrecision(tp,fp,npos)

  -- compute precision/recall
  fp=torch.cumsum(fp)
  tp=torch.cumsum(tp)

  local rec=tp/npos
  local prec=torch.cdiv(tp, (fp+tp+1e-16))

  -- compute average precision
  local ap= torch.zeros(1)
  for t=0,1,0.1 do
    local tmp = prec[rec:ge(t)]
    local p= 0
    if tmp:nDimension() > 0 then
      p= torch.max(tmp)
    end

    if p < 1 then
      p=0
    end
    ap=ap+p/11
  end
  return rec,prec,ap
end


local function xVOCap(rec,prec)
  -- From the PASCAL VOC 2011 devkit
  local mrec=torch.cat(torch.zeros(1), rec):cat(torch.ones(1))
  local mpre=torch.cat(torch.zeros(1), prec):cat(torch.zeros(1))

  for i=mpre:size(1)-1,1,-1 do
    mpre[i]=math.max(mpre[i],mpre[i+1])
  end

  local indexA = torch.ByteTensor(mrec:size()):zero()
  local indexB = torch.ByteTensor(mrec:size()):zero()
  for ii = 2,mrec:size(1) do
    if mrec[ii-1]~=mrec[ii] then
      indexA[ii] = 1
      indexB[ii-1] = 1
    end
  end

  local ap=torch.sum((mrec[indexA]-mrec[indexB]):cmul(mpre[indexB]))

  return ap
end


local function evaluateTpFp(matches,gt)
  local iou,tp,fp,npos = 0, 0, 0, 0
  npos = #gt.rois
  --iterate over all matches and determain ioU
  for i,m in ipairs(matches) do
    local roi_m
    if m.p > 0.9 then
      roi_m = m.r
    else
      goto continue
    end
    for igt,vgt in ipairs(gt) do
      iou = Rect.IoU(vgt.rect,roi_m)

      if iou > 0.5 then --check overlay
        --discriminate between tp and fp
        if vgt.class_index == m.class then --compare class label
          tp = tp + 1
      else
        fp = fp + 1
      end
      end
    end
    ::continue::
  end
  return tp, fp, npos
end


function plot_training_progress(prefix, stats)
  local fn_p = string.format('%s/%sproposal_progress.png',opt.resultDir,prefix)
  local fn_d = string.format('%s/%sdetection_progress.png',opt.resultDir,prefix)
  gnuplot.pngfigure(fn_p)
  gnuplot.title('Traning progress over time (proposal)')

  local xs = torch.range(1, #stats.pcls)

  gnuplot.plot(
    { 'preg', xs, torch.Tensor(stats.preg), '-' },
    { 'pcls', xs, torch.Tensor(stats.pcls), '-' }
  )

  gnuplot.axis({ 0, #stats.pcls, 0, 10 })
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')

  gnuplot.pngfigure(fn_d)
  gnuplot.title('Traning progress over time (detection)')

  gnuplot.plotflush()
  gnuplot.plot(
    { 'dreg', xs, torch.Tensor(stats.dreg), '-' },
    { 'dcls', xs, torch.Tensor(stats.dcls), '-' }
  )

  gnuplot.axis({ 0, #stats.pcls, 0, 10 })
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')

  gnuplot.plotflush()
end


function load_model1(cfg, model_path, network_filename, cuda)

  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg)
  graph.dot(model.pnet.fg, 'pnet',string.format('%s/pnet_fg',opt.resultDir))
  graph.dot(model.pnet.bg, 'pnet',string.format('%s/pnet_bg',opt.resultDir))
  graph.dot(model.cnet.fg, 'cnet', string.format('%s/cnet_fg',opt.resultDir))
  graph.dot(model.cnet.bg, 'cnet', string.format('%s/cnet_bg',opt.resultDir))

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


function load_model2(cfg, model_path, network_filename, cuda)

  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg)
  graph.dot(model.pnet.fg, 'pnet',string.format('%s/pnet_fg',opt.resultDir))
  graph.dot(model.pnet.bg, 'pnet',string.format('%s/pnet_bg',opt.resultDir))
  graph.dot(model.cnet.fg, 'cnet', string.format('%s/cnet_fg',opt.resultDir))
  graph.dot(model.cnet.bg, 'cnet', string.format('%s/cnet_bg',opt.resultDir))

  if cuda then
    model.cnet:cuda()
    model.pnet:cuda()
  end
  local weights = {}
  local gradient
  -- combine parameters from pnet and cnet into flat tensors
  weights[1], gradient = combine_and_flatten_parameters(model.pnet, model.cnet)
  local training_stats
  if network_filename and #network_filename > 0 then
    local stored = load_obj(network_filename)
    --training_stats = stored.stats
    weights[1]:copy(stored.weights)
  end
  if opt.mode == 'onlyCnet' then
    weights[2], gradient = combine_and_flatten_parameters(model.cnet)
    return model, weights[2], gradient, training_stats
  elseif opt.mode == 'onlyPnet' then
    weights[2], gradient = combine_and_flatten_parameters(model.pnet)
    return model, weights[2], gradient, training_stats
  elseif opt.mode == 'both' then
    return model, weights[1], gradient, training_stats
  end
  
end


function graph_training(cfg, model_path, snapshot_prefix, training_data_filename, network_filename)
  local training_data = load_obj(training_data_filename)
  local file_names = keys(training_data.ground_truth)
  print(string.format("Training data loaded. Dataset: '%s'; Total files: %d; classes: %d; Background: %d)",
    training_data.dataset_name,
    #file_names,
    #training_data.class_names,
    #training_data.background_files))

  -- create/load model
  local model, weights, gradient, training_stats = load_model2(cfg, model_path, network_filename, true)
  if not training_stats then
    training_stats = { pcls={}, preg={}, dcls={}, dreg={} }
  end

  local pnet_copy = nil
  if opt.mode == 'onlyCnet' then
    training_stats = { pcls={}, preg={}, dcls={}, dreg={} }
    pnet_copy = model.pnet
  end

  local batch_iterator = BatchIterator.new(model, training_data)
  local eval_objective_grad = create_objective(model, weights, gradient, batch_iterator, training_stats, confusion_pcls, confusion_ccls, opt.mode, pnet_copy)

  print '==> configuring optimizer'
  local optimState, optimMethod, learnSchedule
  if opt.opti == 'CG' then
    optimState = {
      maxIter = 10000
    }
    optimMethod = optim.cg

  elseif opt.opti == 'LBFGS' then
    optimState = {
      learningRate = opt.lr,
      maxIter = 10000,
      nCorrection = 10
    }
    optimMethod = optim.lbfgs

  elseif opt.opti == 'sgd' then
    optimState = {
      learningRate = opt.lr,
      weightDecay = 1E-5,
      momentum = 0.8, --0.6,
      nesterov = true,
      learningRateDecay = 0,
      dampening = 0.0
    }
    optimMethod = optim.sgd
    --[[
    learnSchedule =
    {
      --  start,     end,     LR,     WD
        {     1,    8000,   5e-2,   5e-4 },
        {  8001,   16000,   1e-2,   1e-4 },
        { 16001,   24000,   5e-3,   5e-5 },
        { 24001,   35000,   1e-4,   1e-5 },
        { 35001,     1e8,   1e-5,      0 }
    }
    ]]
    learnSchedule =
      {
        --  start,     end,     LR,     WD
        {     1,    8000,   5e-4,   5e-5 },
        {  8001,   16000,   1e-4,   1e-5 },
        { 16001,   24000,   1e-5,      0 },
        { 24001,   35000,   1e-5,      0 },
        { 35001,     1e8,   1e-5,      0 }
      }
  elseif opt.opti == 'rmsprop' then
    optimState = {
      learningRate = opt.lr,
      alpha = opt.rms_decay,
      epsilon = 1E-3
    }
    optimMethod = optim.rmsprop

  elseif opt.opti == 'adagrad' then
    optimState = {
      learningRate = opt.lr,
      learningRateDecay = 0,
      weightDecay = 0
    }
    optimMethod = optim.adagrad

  elseif opt.opti == 'adam' then
    optimState = {
      learningRate = opt.lr,
      beta1 = 0.9,      -- first moment coefficient
      beta2 = 0.999     -- second moment coefficient
    }
    optimMethod = optim.adam

  elseif opt.opti == 'adadelta' then
    optimState = {
      rho = 0.9, -- interpolation parameter
      weightDecay = 0
    }
    optimMethod = optim.adadelta

  else
    error('unknown optimization method')
  end

  for i=1,50000 do

    if opt.mode == 'onlyCnet' then
      if i == 100 then
        opt.lr = 1e-6
        optimState.learningRate = opt.lr
      end
    end

    if (i % 500) == 0 then
      opt.lr = opt.lr - opt.lr/2
      optimState.learningRate = opt.lr
    end

    if opt.opti == 'sgd' then
      for _,row in ipairs(learnSchedule) do
        if i >= row[1] and i <= row[2] then
          optimState.learningRate = row[3]
          optimState.weightDecay = row[4]
        end
      end
    end

    local timer = torch.Timer()
    local _, loss = optimMethod(eval_objective_grad, weights, optimState)

    local time = timer:time().real

    print('---------------------------------------------------------------------------------')
    print(string.format('[Main:graph_training] %d: loss: %f', i, loss[1]))
    print('---------------------------------------------------------------------------------')

    if i%opt.plot == 0 then
      confusion_pcls:updateValids()
      confusion_ccls:updateValids()
      local train_acc_pnet = confusion_pcls.totalValid * 100
      local train_acc_cnet = confusion_ccls.totalValid * 100
      print(string.format('[Main:graph_training] Train accuracy: pnet'..c.cyan'%.2f cnet:'..c.cyan'%.2f ',train_acc_pnet,train_acc_cnet))
      print(string.format('training pnet confusion: %s',tostring(confusion_pcls)))
      if opt.mode ~= 'onlyPnet' then
        print(string.format('training cnet confusion: %s',tostring(confusion_ccls)))
      end
      plot_training_progress(snapshot_prefix, training_stats)
      evaluation( model, training_data, optimState, i)

      confusion_pcls:zero()
      confusion_ccls:zero()
    end

    if i%opt.snapshot == 0 then
      -- save snapshot
      save_model(string.format('%s/%s_%06d.t7', opt.resultDir,snapshot_prefix, i), weights, opt, training_stats)
    end

  end

  -- compute positive anchors, add anchors to ground-truth file
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

function evaluation(model, training_data,optimState,epoch)
  local batch_iterator = BatchIterator.new(model, training_data)

  local red = torch.Tensor({1,0,0})
  local green = torch.Tensor({0,1,0})
  local blue = torch.Tensor({0,0,1})
  local white = torch.Tensor({1,1,1})
  local colors = { red, green, blue, white }

  -- create detector
  local d = Detector(model,opt.mode)
  local npos = 0
  local tp = torch.zeros(20)
  local fp = torch.zeros(20)
  local save = opt.resultDir

  for i=1,20 do
    --print(string.format('[Main:evaluation] iteration: %d',i))
    -- pick random validation image
    local b = batch_iterator:nextValidation(1)[1]
    local img = b.img:cuda()
    local matches = d:detect(img)
    local v

    if opt.mode ~= 'onlyPnet' then
      tp[i],fp[i],v = evaluateTpFp(matches,b)
      npos = npos + v
    end

    if color_space == 'yuv' then
      img = image.yuv2rgb(img)
    elseif color_space == 'lab' then
      img = image.lab2rgb(img)
    elseif color_space == 'hsv' then
      img = image.hsv2rgb(img)
    end
    -- draw bounding boxes and save image
    for i,m in ipairs(matches) do
      if m.class == 201 then
        draw_rectangle(img, m.r,string.format("CI: %d",m.class), red)
      else
        draw_rectangle(img, m.r,string.format("CI: %d",m.class), green)
      end
    end
    for ii = 1,#b.rois do
      draw_rectangle(img, b.rois[ii].rect,string.format("CI: %d",b.rois[ii].class_index), white)
    end

    image.saveJPG(string.format('%s/output%d.jpg',save, i), img)
  end -- for i=1,20 do

  if opt.mode ~= 'onlyPnet' then
    local rec, prec, ap = averagePrecision(tp,fp,npos)
    ap = xVOCap(rec,prec)
    print(string.format("mAP : %04f",ap*100))
  end

  local base64im_p
  local base64im_d
  do
    os.execute(('openssl base64 -in %s/%sproposal_progress.png -out %s/%s_proposal_progress.base64'):format(save,opt.name,save,opt.name))
    os.execute(('openssl base64 -in %s/%sdetection_progress.png -out %s/%s_detection_progress.base64'):format(save,opt.name,save,opt.name))
    local f_p = io.open(string.format('%s/%s_proposal_progress.base64',save,opt.name))
    local f_d = io.open(string.format('%s/%s_detection_progress.base64',save,opt.name))
    if f_p then base64im_p = f_p:read'*all' end
    if f_d then base64im_d = f_d:read'*all' end
  end
  local file = io.open(string.format('%s/report.html',save),'w')
  if file == nil then
    print(string.format('Unable to read %s/report.html!!!',save))
  else
    file:write(string.format([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]],save,epoch,base64im_p,base64im_d))
    if optimState then
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end

    end
    file:write'<table>\n'
    for i =1,20 do
      file:write(string.format('<tr><img src="output%d.jpg" alt="output" width="244" height="244" ></tr>\n',i))
    end
    file:write'</table>\n'
    file:write'<pre>\n'
    file:write'Training pcls\n'
    file:write(tostring(confusion_pcls)..'\n')
    file:write'</pre>\n'
    file:write'<pre>\n'
    file:write'Training ccls\n'
    file:write(tostring(confusion_ccls)..'\n')
    file:write'</pre>\n'
    file:write(string.format('<td>%s<img src="%s.svg" alt="%s" width="300" height="600" ></td>\n','cnet_fg','cnet_fg','cnet_fg'))
    file:write(string.format('<td>%s<img src="%s.svg" alt="%s" width="300" height="600" ></td>\n','cnet_bg','cnet_bg','cnet_bg'))
    file:write(string.format('<td>%s<img src="%s.svg" alt="%s" width="300" height="600" ></td>\n','pnet_fg','pnet_fg','pnet_fg'))
    file:write(string.format('<td>%s<img src="%s.svg" alt="%s" width="300" height="600" ></td>\n','pnet_bg','pnet_bg','pnet_bg'))
    file:write'</body></html>'
    file:close()
  end

end


graph_training(cfg, opt.model, opt.name, opt.train, opt.restore)
--evaluation_demo(cfg, opt.model, opt.train, opt.restore)
