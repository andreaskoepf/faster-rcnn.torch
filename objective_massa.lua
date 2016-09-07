require 'cunn'
require 'BatchIterator'
require 'Localizer'
require 'ROIPoolingMassa'

function extract_roi_pooling_input(input_rect, localizer, feature_layer_output)
  local r = localizer:inputToFeatureRect(input_rect)
  -- the use of math.min ensures correct handling of empty rects,
  -- +1 offset for top, left only is conversion from half-open 0-based interval
  local s = feature_layer_output:size()
  r = r:clip(Rect.new(0, 0, s[4], s[3]))
  r = r:snapToInt()
  local idx = { }
  if r:isEmpty() then
    --print("rect is empty for feature map")
    return nil, idx
  end
  local idx = { 1, {}, { math.min(r.minY + 1, r.maxY), r.maxY }, { math.min(r.minX + 1, r.maxX), r.maxX } }
  return feature_layer_output[idx], idx
end

function create_objective(model, weights, gradient, batch_iterator, stats, pnet_confusion, cnet_confusion, mode, pnet_copy)
  local cfg = model.cfg
  local pnet = model.pnet
  local cnet = model.cnet

  local bgclass = cfg.backgroundClass or cfg.class_count + 1   -- background class
  local anchors = batch_iterator.anchors
  local localizer = Localizer.new(pnet.outnode.children[#pnet.outnode.children])

  local crossEntropy = nn.CrossEntropyCriterion():cuda()
  local smoothL1 = nn.SmoothL1Criterion():cuda()
  smoothL1.sizeAverage = false
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local cnet_input_planes = model.layers[#model.layers].filters

  local amp = nn.SpatialAdaptiveMaxPooling(kw, kh):cuda()
  local roiPooling = nn.ROIPoolingMassa(kw,kh):cuda()

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
    local delta_outputs_anchors = {}
    local delta_outputs_featureMap

    -- statistics for fine-tuning and classification stage
    local creg_loss, creg_count = 0, 0
    local ccls_loss, ccls_count = 0, 0

    -- enable dropouts
    if pnet_copy then
      pnet_copy:evaluate()
    end
    pnet:training()
    cnet:training()

    local roiPoolInTable1 = {}
    local roiPoolInTable2 = {}
    local groundTruthROIsAndLabels = {}
    local flagTable = {}

    local batch = batch_iterator:nextTraining()

    print('Loop over all images within current batch:')
    print('==========================================')
    for i,x in ipairs(batch) do
      local img = x.img:cuda()    -- convert batch to cuda if we are running on the gpu
      local p = x.positive        -- get positive and negative anchors examples
      local n = x.negative
      local imgID = i

      print(string.format('Working with image: %d', i))

      -- run forward convolution
      local outputs = pnet:forward(img:view(1, img:size(1), img:size(2), img:size(3)))
      if mode == 'onlyCnet' then
        fixedPnetOutputs = pnet_copy:forward(img:view(1, img:size(1), img:size(2), img:size(3)))
      end

      -- ensure all example anchors lie withing existing feature planes
      if mode ~= 'onlyCnet' then
        cleanAnchors(n, outputs)
        cleanAnchors(p, outputs)
      else
        cleanAnchors(n, fixedPnetOutputs)
        cleanAnchors(p, fixedPnetOutputs)
      end

      -- clear delta values for each new image
      for i,out in ipairs(outputs) do
        if not delta_outputs_anchors[i] then
          delta_outputs_anchors[i] = torch.FloatTensor():cuda()
        end
        delta_outputs_anchors[i]:resizeAs(out)
        delta_outputs_anchors[i]:zero()
      end

      -- ROI Pooling input calculation:
      -- ==============================
      table.insert(roiPoolInTable1, { imgID = imgID, featureMap = outputs[#outputs] })
      local input_size = img:size()
      local cnetgrad

      local target = torch.Tensor({{1,0}})

      -- process positive set
      for i,x in ipairs(p) do
        local anchor = x[1]
        local roi = x[2]
        local l = anchor.layer

        local out
        if mode ~= 'onlyCnet' then
          out = outputs[l]
        else
          out = fixedPnetOutputs[l]
        end
        local delta_out = delta_outputs_anchors[l]

        local idx = anchor.index
        local v = out[idx]
        local d = delta_out[idx]

        -- foreground/background classification
        if mode ~= 'onlyCnet' then
          cls_loss = cls_loss + crossEntropy:forward(v[{{1, 2}}], 1)
          local dc = crossEntropy:backward(v[{{1, 2}}], 1)
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

        -- extract roi pooling input
        if mode ~= 'onlyPnet' then
          local reg_proposal = Anchors.anchorToInput(anchor, reg_out) --reg_out
          local pi, idx = extract_roi_pooling_input(reg_proposal, localizer, outputs[#outputs])
          if pi then
            local propRectFM = Rect.new(idx[4][1], idx[3][1], idx[4][2], idx[3][2]) -- minX, minY, maxX, maxY
            table.insert(roiPoolInTable2, { imgID = imgID,
                                            minX = propRectFM.minX, 
                                            minY = propRectFM.minY, 
                                            maxX = propRectFM.maxX, 
                                            maxY = propRectFM.maxY })
            table.insert(groundTruthROIsAndLabels, { roi = roi, label = roi.class_index, proposalROI = reg_proposal })
            table.insert(flagTable, 'positive')
          end
        end
      end

      target = torch.Tensor({{0,1}})

      -- process negative
      for i,x in ipairs(n) do
        local anchor = x[1]
        local l = anchor.layer
        local out
        if mode ~= 'onlyCnet' then
          out = outputs[l]
        else
          out = fixedPnetOutputs[l]
        end
        local delta_out = delta_outputs_anchors[l]
        local idx = anchor.index
        local v = out[idx]
        local d = delta_out[idx]

        if mode ~= 'onlyCnet' then
          cls_loss = cls_loss + crossEntropy:forward(v[{{1, 2}}], 2)
          local dc = crossEntropy:backward(v[{{1, 2}}], 2)
          d[{{1,2}}]:add(dc)
        end

        pnet_confusion:batchAdd(v[{{1, 2}}]:reshape(1,2), target)

        -- extract roi pooling input
        if mode ~= 'onlyPnet' then
          local outpnetCLS = v[{{1, 2}}]:reshape(1,2)
          if outpnetCLS[1][1] > 0.7 then -- take negatives if and only if pnet has classified them as positive
            --local reg_out = v[{{3, 6}}] -- Anchor  
            --local reg_proposal = Anchors.anchorToInput(anchor, reg_out) --reg_out
            --local pi, idx = extract_roi_pooling_input(reg_proposal, localizer, outputs[#outputs])
            local pi, idx = extract_roi_pooling_input(anchor, localizer, outputs[#outputs])
            if pi then
              local propRectFM = Rect.new(idx[4][1], idx[3][1], idx[4][2], idx[3][2]) -- minX, minY, maxX, maxY
              table.insert(roiPoolInTable2, { imgID = imgID, 
                                              minX = propRectFM.minX, 
                                              minY = propRectFM.minY, 
                                              maxX = propRectFM.maxX, 
                                              maxY = propRectFM.maxY })
              table.insert(groundTruthROIsAndLabels, { roi = nil, label = bgclass, proposalROI = nil })
              table.insert(flagTable, 'negative')
            end -- if pi then
          end -- if outpnetCLS[1][1] > 0.7 then
        end -- if mode ~= 'onlyPnet' then
      end -- for i,x in ipairs(n) do

      table.insert(delta_outputs, delta_outputs_anchors)

      -- Terms for gradient normalization:
      reg_count = reg_count + #p
      cls_count = cls_count + #p + #n
      creg_count = creg_count + #p
      ccls_count = ccls_count + 1

    end -- for i,x in ipairs(batch) do

    -- pass through adaptive max pooling operation by F. Massa:
    print("#images with BBoxes:")
    print(#batch)
    print('#roiPoolInTable1:')
    print(#roiPoolInTable1)
    print('#roiPoolInTable2:')
    print(#roiPoolInTable2)
    local roiPoolingInput1 = torch.CudaTensor(#batch, 
                                              roiPoolInTable1[1].featureMap:size(2), 
                                              roiPoolInTable1[1].featureMap:size(3),
                                              roiPoolInTable1[1].featureMap:size(4))
    for i=1,#roiPoolInTable1 do
      roiPoolingInput1[i] = roiPoolInTable1[i].featureMap:view(roiPoolInTable1[i].featureMap:size(2), 
                                                               roiPoolInTable1[i].featureMap:size(3), 
                                                               roiPoolInTable1[i].featureMap:size(4))
    end
    print('roiPoolingInput1:size():')
    print(roiPoolingInput1:size())
    local roiPoolingInput2 = torch.CudaTensor(#roiPoolInTable2, 5)
    local imageId = 1
    local minX, minY, maxX, maxY = 0, 0, 0, 0
    for i=1,#roiPoolInTable2 do
      imageId = roiPoolInTable2[i].imgID
      minX    = roiPoolInTable2[i].minX
      minY    = roiPoolInTable2[i].minY
      maxX    = roiPoolInTable2[i].maxX
      maxY    = roiPoolInTable2[i].maxY
      roiPoolingInput2[{i,{}}] = torch.CudaTensor({imageId, minX, minY, maxX, maxY})
    end
    print('roiPoolingInput2:size():')
    print(roiPoolingInput2:size())
    local inputTable = {roiPoolingInput1, roiPoolingInput2}
    print('inputTable:')
    print(inputTable)
    print('Perform roi pooling by F. Massa for all ROIs and all images:')
    print('============================================================')
    local roiPoolingOutputMassa = roiPooling:forward(inputTable)
    print('roiPoolingOutputMassa:size():')
    print(roiPoolingOutputMassa:size())

    -- ========================================================

    -- fine-tuning STAGE
    print('FINE-TUNING STAGE')
    print('=================')
    -- pass extracted roi-data through classification network
    if mode ~= 'onlyPnet' then

      -- create cnet input batch
      if roiPoolingOutputMassa:size(1) > 0 then

        local cinput = torch.CudaTensor(roiPoolingOutputMassa:size(1), kh * kw * cnet_input_planes)
        local cctarget = torch.CudaTensor(roiPoolingOutputMassa:size(1)):zero()
        local crtarget = torch.CudaTensor(roiPoolingOutputMassa:size(1), 4):zero()

        for i=1,roiPoolingOutputMassa:size(1) do
          cinput[i] = roiPoolingOutputMassa[i]
          if groundTruthROIsAndLabels[i].roi then
            -- positive example
            cctarget[i] = groundTruthROIsAndLabels[i].label
            local groundTruthROI = groundTruthROIsAndLabels[i].roi
            local proposalROI = groundTruthROIsAndLabels[i].proposalROI
            crtarget[i] = Anchors.inputToAnchor(proposalROI, groundTruthROI.rect)
          else
            -- negative example
            cctarget[i] = bgclass
          end
        end

        -- process classification batch
        local coutputs = cnet:forward(cinput)

        -- compute classification and regression error and run backward pass
        lambda = 10
        local crout = coutputs[1]
        print('crout:size():')
        print(crout:size())
        for i=1,#flagTable do
          if flagTable[i] == 'negative' then -- ignore negative examples
            crout[{i, {}}]:zero()
          end
        end
        creg_loss = creg_loss + smoothL1:forward(crout, crtarget)* lambda -- * 10
        local crdelta = smoothL1:backward(crout, crtarget) * lambda

        local ccout = coutputs[2]

        local loss = crossEntropy:forward(ccout, cctarget)
        ccls_loss = ccls_loss + loss
        local ccdelta = crossEntropy:backward(ccout, cctarget)

        --print('Is cinput zero?')
        --print(torch.all(torch.eq(cinput, 0)))
        --print('Is crdelta zero?')
        --print(torch.all(torch.eq(crdelta, 0)))
        --print('Is ccdelta zero?')
        --print(torch.all(torch.eq(ccdelta, 0)))

        local post_roi_delta = cnet:backward(cinput, { crdelta, ccdelta })

        --print('Is post_roi_delta zero?')
        --print(torch.all(torch.eq(post_roi_delta, 0)))

        cnet_confusion:batchAdd(ccout, cctarget)
        post_roi_delta = post_roi_delta:view(roiPoolingOutputMassa:size(1), cnet_input_planes, kh, kw) -- first dimension is #rois
        post_roi_delta = post_roi_delta:cuda()
        delta_outputs_featureMap = roiPooling:backward(inputTable, post_roi_delta)[1]
        --print('delta_outputs_featureMap:size()')
        --print(delta_outputs_featureMap:size())

      end -- if #roi_pool_state > 0
    end -- if mode ~= 'onlyPnet'

    -- backward pass of proposal network
    for i,x in ipairs(batch) do
      local img = x.img:cuda()
      delta_outputs[i][#delta_outputs[i]] = delta_outputs_featureMap[i]:view(1, 
                                                                             delta_outputs_featureMap:size(2),
                                                                             delta_outputs_featureMap:size(3),
                                                                             delta_outputs_featureMap:size(4))
      --print(string.format('delta_outputs[%d]:', i))
      --print(delta_outputs[i])
      local gi = pnet:backward(img, delta_outputs[i])
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

    print(string.format('prop: cls: %f (%d), reg: %f (%d); det: cls: %f (%d), reg: %f (%d)',
      pcls, cls_count, preg, reg_count, dcls, ccls_count, dreg, creg_count)
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
