require 'cunn'
require 'BatchIterator'
require 'Localizer'

function createObjective(pnet, cnet, weights, gradient, batch_iterator)
  local training_data = batch_iterator.trainingData
  local anchors = batch_iterator.anchors    
  local localizer = Localizer.new(pnet.outnode.children[5])
    
  local softmax = nn.CrossEntropyCriterion():cuda()
  local cnll = nn.ClassNLLCriterion():cuda()
  local smoothL1 = nn.SmoothL1Criterion():cuda()
  smoothL1.sizeAverage = false
  local kh, kw = trainingData.roi_pooling.kh, trainingData.roi_pooling.kw
  local amp = nn.SpatialAdaptiveMaxPooling(kw, kh):cuda()
  
  function lossAndGradient(w)
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
        
      end
      
      while cls_count < 256 do
      
      
        
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
    
    return lossAndGradient
end
