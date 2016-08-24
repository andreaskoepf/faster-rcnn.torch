require 'cunn'
require 'BatchIterator'
require 'Localizer'

function extract_roi_pooling_input(input_rect, localizer, feature_layer_output)
  --print(input_rect)
  local r = localizer:inputToFeatureRect(input_rect)
  -- the use of math.min ensures correct handling of empty rects,
  -- +1 offset for top, left only is conversion from half-open 0-based interval
  local s = feature_layer_output:size()
  --print(r)
  --print(s)
  --io.read()
  r = r:clip(Rect.new(0, 0, s[3], s[2]))
  
  local idx = { {}, { math.min(r.minY + 1, r.maxY), math.max(r.minY + 1, r.maxY) }, { math.min(r.minX + 1, r.maxX),  math.max(r.minX + 1, r.maxX) } }
  return feature_layer_output[idx], idx
end

function create_objective(model, weights, gradient, batch_iterator, stats,pnet_confusion,cnet_confusion)
  print("==> create objective")
  local cfg = model.cfg
  local fnet = model.fnet
  local pnet = model.pnet
  local cnet = model.cnet
  collectgarbage()
  --local bgclass = cfg.class_count + 1   -- background class
  
  local bgclass = cfg.backgroundClass or cfg.class_count +1   -- background class
  local anchors = batch_iterator.anchors
  --local localizer = Localizer.new(pnet.outnode.children[#pnet.outnode.children])
  local localizer = Localizer.new(fnet.outnode)
  
  local softmax = nn.CrossEntropyCriterion():cuda()
  local cnll = nn.ClassNLLCriterion():cuda()
  local smoothL1 = nn.SmoothL1Criterion():cuda()
  --local smoothL1 = nn.MSECriterion():cuda()
  smoothL1.sizeAverage = false
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local cnet_input_planes = model.layers[#model.layers].filters
  local amp = nn.SpatialAdaptiveMaxPooling(kw, kh):cuda()

  local function cleanAnchors(examples, outputs)
    local i = 1
    while i <= #examples do
      local anchor = examples[i][1]
      
      --print(outputs)
      local fmSize = outputs:size()
      if anchor.index[2] > fmSize[2] or anchor.index[3] > fmSize[3] then
        table.remove(examples, i)   -- accessing would cause ouf of range exception
      else
        i = i + 1
      end
    end
  end
  
  local function lossAndGradient(w)
    if w ~= weights then
      weights:copy(w)
    end
    gradient:zero()


    -- statistics for proposal stage
    local cls_loss, reg_loss = 0, 0
    local cls_count, reg_count = 0, 0
    local delta_outputs = {}

    -- statistics for fine-tuning and classification stage
    local creg_loss, creg_count = 0, 0
    local ccls_loss, ccls_count = 0, 0

    print("==> switch all to training")
    -- enable dropouts
    fnet:training()
    pnet:training()
    cnet:training()
    print("==> switched all to training")
    
    local batch = batch_iterator:nextTraining()
    print("==> batch received")
    local target = torch.Tensor({{1,0}})
    local clsOutput = torch.Tensor({{1,0}})
    --gradient:zero()
    for i,x in ipairs(batch) do
      print(string.format("==> batch iteration: %d",i))
      local img = x.img:cuda()    -- convert batch to cuda if we are running on the gpu
      local p = x.positive        -- get positive and negative anchors examples
      local n = x.negative
      -- run forward convolution
      print("features")
      local features = fnet:forward(img)
      
      local outputs = pnet:forward(features)
      -- ensure all example anchors lie withing existing feature planes
      cleanAnchors(n, outputs)
      cleanAnchors(p, outputs)
      print(outputs)
      -- clear delta values for each new image
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

      target = torch.Tensor({{1,0}})
      
      local lambda = 5
      -- process positive set
      for i,x in ipairs(p) do
        local anchor = x[1]
        local roi = x[2]
        local l = anchor.layer

        local out = outputs[l]
        local delta_out = delta_outputs[l]
        local idx = anchor.index--anchor.index
        local v = out[idx]
        local d = delta_out[idx]

        -- foreground/background classification
        local res = v[{{1, 2}}]
        cls_loss = cls_loss + softmax:forward(res, torch.ones(res:size()):cuda())
        local dc = softmax:backward(res, torch.ones(res:size()):cuda())
        d[{{1,2}}]:add(dc)
        clsOutput[{1,1}]=v[1]
        clsOutput[{1,2}]=v[2]
        pnet_confusion:batchAdd(clsOutput, target)

        -- box regression
        local reg_out = v[{{3, 6}}] -- Anchor
        local reg_target = Anchors.inputToAnchor(anchor, roi.rect):cuda()  -- regression target

        reg_loss = reg_loss + smoothL1:forward(reg_out, reg_target) * lambda-- * 10
        local dr = smoothL1:backward(reg_out, reg_target) * lambda --* 10
        d[{{3,6}}]:add(dr)
        
        -- pass through adaptive max pooling operation
        local pi, idx = extract_roi_pooling_input(roi.rect, localizer, outputs[#outputs])
        local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
        local reg_proposal = Anchors.anchorToInput(anchor,reg_out)-- roi.rect--Rect.new(reg_out[1],reg_out[2],reg_out[3],reg_out[4])--
        table.insert(roi_pool_state, { input = pi, input_idx = idx, anchor = anchor, reg_proposal = reg_proposal, roi = roi, output = po:clone(), indices = amp.indices:clone() })
      end
      
      target = torch.Tensor({{0,1}})
      -- process negative
      for i,x in ipairs(n) do
        local anchor = x[1]
        local l = anchor.layer
        local out = outputs[l]
        local delta_out = delta_outputs[l]
        local idx = anchor.index
        local v = out[idx]
        local d = delta_out[idx]

        local res = v[{{1, 2}}]
        local clstarg = torch.ones(res:size())+torch.ones(res:size())
        cls_loss = cls_loss + softmax:forward(res, clstarg:cuda())
        local dc = softmax:backward(res, clstarg:cuda())
        d[{{1,2}}]:add(dc)
        clsOutput[{1,1}]=v[1]
        clsOutput[{1,2}]=v[2]

        pnet_confusion:batchAdd(clsOutput, target)
        -- pass through adaptive max pooling operation
        local pi, idx = extract_roi_pooling_input(anchor, localizer, outputs[#outputs])
        local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
        table.insert(roi_pool_state, { input = pi, input_idx = idx, output = po:clone(), indices = amp.indices:clone() })
      end

      -- fine-tuning STAGE
      -- pass extracted roi-data through classification network

      -- create cnet input batch
      if #roi_pool_state > 0 then
        local cinput = torch.CudaTensor(#roi_pool_state, kh * kw * cnet_input_planes)
        local cctarget = torch.CudaTensor(#roi_pool_state)
        local crtarget = torch.CudaTensor(#roi_pool_state, 4):zero()

        for i,x in ipairs(roi_pool_state) do
          cinput[i] = x.output
          if x.roi then
            -- positive example
            cctarget[i] = x.roi.class_index
            crtarget[i] = Anchors.inputToAnchor(x.reg_proposal, x.roi.rect)   -- base fine tuning on proposal
          else
            -- negative example
            cctarget[i] = bgclass
          end
          --delta_outputs[i]:resizeAs(out)
          --delta_outputs[i]:zero()
        end

        -- process classification batch
        local coutputs = cnet:forward(cinput)
        -- compute classification and regression error and run backward pass
        
        local crout = coutputs[1]
        --print("Output of coutputs")
        --print(crout)
        --print(string.format("Positive anchors: %d , Number of roi pool state: %d",#p,#roi_pool_state))
        if #p<#roi_pool_state then
            crout[{{#p + 1, #roi_pool_state}, {}}]:zero() -- ignore negative examples
        end
        creg_loss = creg_loss + smoothL1:forward(crout, crtarget)
        local crdelta = smoothL1:backward(crout, crtarget)
        
        local ccout = coutputs[2]  -- log softmax classification
        
        --criterion ClassNLL
        local loss = cnll:forward(ccout, cctarget)
        ccls_loss = ccls_loss + loss
        --print(string.format("Ccls_loss: %04f",ccls_loss))
        local ccdelta = cnll:backward(ccout, cctarget)


        --[[ ccls_loss = ccls_loss + softmax:forward(ccout, cctarget)
        local ccdelta = softmax:backward(ccout, cctarget)--]]

        --local post_roi_delta = cnet:backward(cinput, { crdelta, ccdelta })

        cnet_confusion:batchAdd(ccout, cctarget)
        -- run backward pass over rois
        --for i,x in ipairs(roi_pool_state) do
        --  amp.indices = x.indices
        --  delta_outputs[#delta_outputs][x.input_idx]:add(amp:backward(x.input, post_roi_delta[i]:view(cnet_input_planes, kh, kw)))
        --end
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
    gradient:div(cls_count+creg_count)
    
    local pcls = cls_loss / cls_count     -- proposal classification (bg/fg)
    local preg = reg_loss / reg_count     -- proposal bb regression
    local dcls = ccls_loss / ccls_count   -- detection classification
    local dreg = creg_loss / creg_count   -- detection bb finetuning

    print(string.format('prop: cls: %f (%d), reg: %f (%d); det: cls: %f, reg: %f',
      pcls, cls_count, preg, reg_count, dcls, dreg)
    )

    table.insert(stats.pcls, pcls)
    table.insert(stats.preg, preg)
    table.insert(stats.dcls, dcls)
    table.insert(stats.dreg, dreg)

    local loss = pcls + preg
    return loss, gradient
  end

  return lossAndGradient
end
