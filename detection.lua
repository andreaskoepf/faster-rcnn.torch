require 'image'
require 'nms'


function detect(img, model)
  
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