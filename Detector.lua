require 'cunn'
require 'image'
require 'nms'
require 'Anchors'

local Detector = torch.class('Detector')

function Detector:__init(model, mode, pnet_copy)
  self.mode = mode or 'both'
  self.pnet_copy = pnet_copy or nil
  local cfg = model.cfg
  self.model = model
  self.anchors = Anchors.new(model.pnet, model.cfg.scales)
  self.localizer = Localizer.new(model.pnet.outnode.children[#model.pnet.outnode.children])
  self.m = nn.SoftMax():cuda()
  self.amp = nn.SpatialAdaptiveMaxPooling(cfg.roi_pooling.kw, cfg.roi_pooling.kh):cuda()
end


function Detector:detectFgBg(input)
  local cfg = self.model.cfg
  local pnet = self.pnet_copy or self.model.pnet
  local m = self.m

  local input_size = input:size()

  local input_rect = Rect.new(0, 0, input_size[3], input_size[2])

  -- pass image through network
  pnet:evaluate()
  input = input:cuda()
  local outputs = pnet:forward(input:view(1, input:size(1), input:size(2), input:size(3)))

  -- analyse network output for non-background classification
  local matches = {}

  -- only analyze anchors which are fully inside the input image
  local ranges = self.anchors:findRangesXY(input_rect, input_rect)

  for i,r in ipairs(ranges) do
    local layer = outputs[r.layer]
    local a = r.aspect
    local ofs = (a-1) * 6

    for y=r.ly,r.uy-1 do
      for x=r.lx,r.ux-1 do

        local cls_out = layer[{1, {ofs + 1, ofs + 2}, y, x}]
        local reg_out = layer[{1, {ofs + 3, ofs + 6}, y, x}]

        -- classification

        --local c_prop = m:forward(cls_out)

        if cls_out[1] > cls_out[2] then  -- only two classes (foreground and background)

          -- regression
          local a_ = self.anchors:get(r.layer, a, y, x)
          local anchor_rect = Anchors.anchorToInput(a_, reg_out)
          if anchor_rect:overlaps(input_rect) then
            table.insert(matches, { p=cls_out[1], a=a_, r=anchor_rect, l=r.layer })
          end
        end

      end
    end

  end
  return matches
end


function Detector:detectObjects(input,matches)
  local cfg = self.model.cfg
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local cnet_input_planes = self.model.layers[#self.model.layers].filters
  local winners = {}
  local bgclass = cfg.backgroundClass or cfg.class_count + 1   -- background class
  local amp = self.amp
  local cnet = self.model.cnet
  local pnet = self.model.pnet


  if #matches > 0 then

    local candidates = matches

    -- NON-MAXIMUM SUPPRESSION

    local bb = torch.Tensor(#matches, 4)
    local score = torch.Tensor(#matches, 1)
    for i=1,#matches do
      bb[i] = matches[i].r:totensor()
      score[i] = matches[i].p
    end

    local iou_threshold = 0.45 --FIXME =0.25
    local pick = nms(bb, iou_threshold, score)
    --local pick = nms(bb, iou_threshold, 'area')
    local candidates = {}
    pick:apply(function (x) table.insert(candidates, matches[x]) end )

    -- REGION CLASSIFICATION
    --cnet:evaluate()
    pnet:evaluate()
    -- create cnet input batch
    local cinput = torch.CudaTensor(#candidates, cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes):zero()
    local outputs = pnet:forward(input:view(1, input:size(1), input:size(2), input:size(3)))
    local counter = 1
    for i,v in ipairs(candidates) do
      -- pass through adaptive max pooling operation
      local pi, idx = extract_roi_pooling_input(v.r, self.localizer, outputs[#outputs])
      if pi then

        local po = amp:forward(pi):view(cfg.roi_pooling.kh * cfg.roi_pooling.kw * cnet_input_planes)

        cinput[counter]:copy(po)
        counter = counter +1
      end
    end
    cinput = cinput[{{1,counter-1},{}}]:clone()
    local coutputs = cnet:forward(cinput)
    -- send extracted roi-data through classification network
    if true then
      winners = self:evaluate(coutputs, candidates)
    else
      winners = self:evaluatewithNMS(coutputs, candidates)
    end
  end

  --print(string.format('[Detector:detect] winners: %d', #winners))
  return winners
end


function Detector:evaluate(coutputs, candidates)
  local bbox_out = coutputs[1]
  local cls_out = coutputs[2]

  local c_norm = self.m:forward(cls_out):clone()
  local yclass = {}

  for i,x in ipairs(candidates) do
    x.r2 = Anchors.anchorToInput(x.r, bbox_out[i])

    local cprob = c_norm[i]

    local p_winner, c_winner = torch.max(cprob,1) -- get max probability and class index

    x.class = c_winner[1] -- c[1]
    x.confidence = p_winner[1] -- p[1]
    if x.confidence > 0.7 then
      table.insert(yclass,x)
    end
  end
  return yclass
end


function Detector:evaluatewithNMS(coutputs, candidates)
  
  local bbox_out = coutputs[1]
  local cls_out = coutputs[2]

  local c_norm = self.m:forward(cls_out):clone()
  local yclass = {}

  for i,x in ipairs(candidates) do
    x.r2 = Anchors.anchorToInput(x.r, bbox_out[i])

    local cprob = c_norm[i]

    local p_winner, c_winner = torch.max(cprob,1) -- get max probability and class index

    x.class = c_winner[1] -- c[1]
    x.confidence = p_winner[1] -- p[1]
    if x.confidence > 0.7 then
      if not yclass[x.class] then
        yclass[x.class] = {}
      end
      table.insert(yclass[x.class],x)
    end
  end

  local winners = {}
  local overlab = 0.5
  -- run per class NMS
  for i,c in pairs(yclass) do
    -- fill rect tensor
    local bb = torch.Tensor(#c, 5)
    for j,r in ipairs(c) do
      bb[{j, {1,4}}] = r.r2:totensor()
      bb[{j, 5}] = r.confidence
    end

    local pick = nms(bb, overlab, bb[{{}, 5}])
    pick:apply(function (x) table.insert(winners, c[x]) end )
  end
  return winners
end


function Detector:detect(input)
  local matches = self:detectFgBg(input)

  if self.mode == 'onlyPnet' then
    return matches
  else
    return self:detectObjects(input,matches)
  end

end
