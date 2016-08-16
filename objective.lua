require 'cunn'
require 'BatchIterator'
require 'Localizer'

function extract_roi_pooling_input(input_rect, localizer, feature_layer_output)
  local r = localizer:inputToFeatureRect(input_rect)
  -- the use of math.min ensures correct handling of empty rects,
  -- +1 offset for top, left only is conversion from half-open 0-based interval
  local s = feature_layer_output:size()
  r = r:clip(Rect.new(0, 0, s[4], s[3]))
  local idx = { 1, {}, { math.min(r.minY + 1, r.maxY), r.maxY }, { math.min(r.minX + 1, r.maxX), r.maxX } }
  return feature_layer_output[idx], idx
end

function create_objective(model, weights, gradient, batch_iterator, stats, pnet_confusion, cnet_confusion, mode, pnet_copy)
  local cfg = model.cfg
  local pnet = model.pnet
  local cnet = model.cnet

  local bgclass = cfg.backgroundClass or cfg.class_count + 1   -- background class
  local anchors = batch_iterator.anchors
  local localizer = Localizer.new(pnet.outnode.children[1])

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
      local fmSize = outputs[anchor.layer]:size()

      if anchor.index[3] > fmSize[3] or anchor.index[4] > fmSize[4] then -- ## added 1 to indices to support batch-normalization
        table.remove(examples, i)   -- accessing would cause ouf of range exception
      else
        i = i + 1
      end
    end
  end


  -- ## temp constant batch for testing
  --local dummy_batch = batch_iterator:nextTraining()

  local lambda = 1    -- weight for box proposal regression

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

    -- enable dropouts
    pnet:training()
    cnet:training()

    local batch = batch_iterator:nextTraining()

    for i,x in ipairs(batch) do
      local img = x.img:cuda()    -- convert batch to cuda if we are running on the gpu
      local p = x.positive        -- get positive and negative anchors examples
      local n = x.negative
      local outputs
      -- run forward convolution
      if mode ~= 'onlyCnet' then
        outputs = pnet:forward(img:view(1, img:size(1), img:size(2), img:size(3)))
      else
        outputs = pnet_copy:forward(img:view(1, img:size(1), img:size(2), img:size(3)))
      end

      -- ensure all example anchors lie withing existing feature planes
      cleanAnchors(n, outputs)
      cleanAnchors(p, outputs)

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

      local target = torch.Tensor({{1,0}})

      -- process positive set
      for i,x in ipairs(p) do
        local anchor = x[1]
        local roi = x[2]
        local l = anchor.layer

        local out = outputs[l]
        local delta_out = delta_outputs[l]

        local idx = anchor.index
        local v = out[idx]
        local d = delta_out[idx]

        -- foreground/background classification
        if mode ~= 'onlyCnet' then
          cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 1)
          local dc = softmax:backward(v[{{1, 2}}], 1)
          d[{{1,2}}]:add(dc)
        end

        pnet_confusion:batchAdd(v[{{1, 2}}]:reshape(1,2), target)

        -- box regression
        local reg_out = v[{{3, 6}}] -- Anchor
        local reg_target = Anchors.inputToAnchor(anchor, roi.rect):cuda()  -- regression target
        if mode ~= 'onlyCnet' then
          reg_loss = reg_loss + smoothL1:forward(reg_out, reg_target) * lambda
          local dr = smoothL1:backward(reg_out, reg_target) * lambda
          d[{{3,6}}]:add(dr)
        end

        -- pass through adaptive max pooling operation
        if mode ~= 'onlyPnet' then
          local pi, idx = extract_roi_pooling_input(roi.rect, localizer, outputs[#outputs])
          local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
          local reg_proposal = Anchors.anchorToInput(anchor, reg_target) --reg_out
          table.insert(roi_pool_state, { input = pi, input_idx = idx, anchor = anchor, reg_proposal = reg_proposal, roi = roi, output = po:clone(), indices = amp.indices:clone() })
        end
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

        if mode ~= 'onlyCnet' then
          cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 2)
          local dc = softmax:backward(v[{{1, 2}}], 2)
          d[{{1,2}}]:add(dc)
        end

        pnet_confusion:batchAdd(v[{{1, 2}}]:reshape(1,2), target)

        -- pass through adaptive max pooling operation
        if mode ~= 'onlyPnet' then
          local pi, idx = extract_roi_pooling_input(anchor, localizer, outputs[#outputs])
          local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
          table.insert(roi_pool_state, { input = pi, input_idx = idx, output = po:clone(), indices = amp.indices:clone() })
        end
      end

      -- fine-tuning STAGE
      -- pass extracted roi-data through classification network
      if mode ~= 'onlyPnet' then

        -- create cnet input batch
        if #roi_pool_state > 0 then

          --if false then
          local cinput = torch.CudaTensor(#roi_pool_state, kh * kw * cnet_input_planes)
          local cctarget = torch.CudaTensor(#roi_pool_state):zero()
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
          lambda = 10
          local crout = coutputs[1]
          crout[{{#p + 1, #roi_pool_state}, {}}]:zero() -- ignore negative examples
          creg_loss = creg_loss + smoothL1:forward(crout, crtarget)* lambda -- * 10
          local crdelta = smoothL1:backward(crout, crtarget) * lambda

          local ccout = coutputs[2]  -- log softmax classification

          --criterion ClassNLL
          local loss = cnll:forward(ccout, cctarget)
          ccls_loss = ccls_loss + loss
          local ccdelta = cnll:backward(ccout, cctarget)

          --[[ ccls_loss = ccls_loss + softmax:forward(ccout, cctarget)
          local ccdelta = softmax:backward(ccout, cctarget)--]]

          local post_roi_delta = cnet:backward(cinput, { crdelta, ccdelta })

          cnet_confusion:batchAdd(ccout, cctarget)
          -- run backward pass over rois
          for i,x in ipairs(roi_pool_state) do
            amp.indices = x.indices
            delta_outputs[#delta_outputs][x.input_idx]:add(amp:backward(x.input, post_roi_delta[i]:view(cnet_input_planes, kh, kw)))
          end
        end -- if #roi_pool_state > 0
      end -- if mode ~= 'onlyPnet'

      -- backward pass of proposal network
      if mode ~= 'onlyCnet' then
        local gi = pnet:backward(img, delta_outputs)
      end

      -- print(string.format('%f; pos: %d; neg: %d', gradient:max(), #p, #n))
      reg_count = reg_count + #p
      cls_count = cls_count + #p + #n
      creg_count = creg_count + #p
      ccls_count = ccls_count + 1
    end -- for i,x in ipairs(batch) do

    -- scale gradient
    if cls_count ~= 0 then
      gradient:div(cls_count) -- will divide all elements of gradient with cls_count in-place
    end

    local pcls = cls_loss    -- proposal classification (bg/fg)
    local preg = reg_loss    -- proposal bb regression
    local dcls = ccls_loss   -- detection classification
    local dreg = creg_loss   -- detection bb finetuning
    if cls_count ~= 0 then
      pcls = pcls / cls_count
    end
    if reg_count ~= 0 then
      preg = preg / reg_count
    end
    if ccls_count ~= 0 then
      dcls = dcls / ccls_count
    end
    if creg_count ~= 0 then
      dreg = dreg / creg_count
    end

    print(string.format('prop: cls: %f (%d), reg: %f (%d); det: cls: %f, reg: %f',
      pcls, cls_count, preg, reg_count, dcls, dreg)
    )

    table.insert(stats.pcls, pcls)
    table.insert(stats.preg, preg)
    table.insert(stats.dcls, dcls)
    table.insert(stats.dreg, dreg)

    local loss = pcls + preg + dcls + dreg

    local absgrad = torch.abs(gradient)
    print(string.format('Gradient: max %f; mean %f', absgrad:max(), absgrad:mean()))

    return loss, gradient
  end

  return lossAndGradient
end
