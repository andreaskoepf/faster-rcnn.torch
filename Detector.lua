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
  self.lsm = nn.LogSoftMax():cuda()
  self.m = nn.SoftMax():cuda()
  self.amp = nn.SpatialAdaptiveMaxPooling(cfg.roi_pooling.kw, cfg.roi_pooling.kh):cuda()
end

function Detector:detect(input)
  local cfg = self.model.cfg
  local pnet = self.pnet_copy or self.model.pnet
  local cnet = self.model.cnet
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local bgclass = cfg.backgroundClass or cfg.class_count + 1   -- background class
  local amp = self.amp
  local lsm = self.lsm
  local m = self.m

  local cnet_input_planes = self.model.layers[#self.model.layers].filters

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
        local c_prop = m:forward(cls_out)

        if c_prop[1] > c_prop[2] then  -- only two classes (foreground and background)

          -- regression
          local a_ = self.anchors:get(r.layer, a, y, x)
          local anchor_rect = Anchors.anchorToInput(a_, reg_out)
          if anchor_rect:overlaps(input_rect) then
            table.insert(matches, { p=c_prop[1], a=a_, r=anchor_rect, l=r.layer })
          end
        end

      end
    end

  end

  local winners = {}

  if self.mode == 'onlyPnet' then
    return matches
  else
    if #matches > 0 then

      local candidates = matches

      -- NON-MAXIMUM SUPPRESSION
      --[[local bb = torch.Tensor(#matches, 4)
      local score = torch.Tensor(#matches, 1)
      for i=1,#matches do
        bb[i] = matches[i].r:totensor()
        score[i] = matches[i].p
      end

      local iou_threshold = 0.25 --FIXME =0.25
      local pick = nms(bb, iou_threshold, score)
      --local pick = nms(bb, iou_threshold, 'area')
      local candidates = {}
      pick:apply(function (x) table.insert(candidates, matches[x]) end )
      --print(string.format('[Detector:detect] candidates: %d', #candidates))
      --]]
      -- REGION CLASSIFICATION
      cnet:evaluate()
      self.model.pnet:evaluate()
      -- create cnet input batch
      local cinput = torch.CudaTensor(#candidates, cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
      outputs = self.model.pnet:forward(input:view(1, input:size(1), input:size(2), input:size(3)))
      for i,v in ipairs(candidates) do
        -- pass through adaptive max pooling operation
        local pi, idx = extract_roi_pooling_input(v.r:snapToInt(), self.localizer, outputs[#outputs])
        if pi then
          local po = amp:forward(pi):view(cfg.roi_pooling.kh * cfg.roi_pooling.kw * cnet_input_planes)
          cinput[i] = po:clone()
        end
      end

      -- send extracted roi-data through classification network
      local coutputs = cnet:forward(cinput)
      local bbox_out = coutputs[1]
      local cls_out = coutputs[2]

      --local c_norm = torch.exp(-1 * cls_out) -- Conversion of LogSoftMax into SoftMax
      local c_norm = m:forward(cls_out)

      local yclass = {}

      for i,x in ipairs(candidates) do
        x.r2 = Anchors.anchorToInput(x.r, bbox_out[i])

        local cprob = c_norm[i]
        local p_winner, c_winner = torch.max(cprob,1) -- get max probability and class index


        x.class = c_winner[1] -- c[1]
        x.confidence = p_winner[1] -- p[1]
        --print(string.format('x.class = %d', x.class))
        --if x.class ~= bgclass and math.exp(x.confidence) > 0.2 then
        --if x.class ~= bgclass and x.confidence > 0.2 then
        --if x.confidence > 0.2 then
         
       --if not yclass[x.class] then
--            yclass[x.class] = {}
          table.insert(yclass,x)
        --else
        --print(x.confidence)
       --end
      end

      --print(string.format('[Detector:detect] yclass: %d', #yclass))
      --print('yclass:')
      --print(yclass)

      local overlab = 0.5
      -- run per class NMS
     --[[
      for i,c in pairs(yclass) do
        -- fill rect tensor
        local bb = torch.Tensor(#c, 5)
        for j,r in ipairs(c) do
          bb[{j, {1,4}}] = r.r2:totensor()
          bb[{j, 5}] = r.confidence
        end

        pick = nms(bb, overlab, bb[{{}, 5}])
        pick:apply(function (x) table.insert(winners, c[x]) end )
      end
--]]
      winners = yclass

    end -- if #matches > 0

    --print(string.format('[Detector:detect] winners: %d', #winners))
    return winners
  end -- if mode ~= 'onlyPnet' then

end
