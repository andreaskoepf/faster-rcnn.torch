require 'cunn'
require 'image'
require 'nms'
require 'Anchors'

local Detector = torch.class('Detector')

function Detector:__init(model)
  local cfg = model.cfg
  self.model = model
  self.anchors = Anchors.new(model.pnet, model.cfg.scales)
  self.localizer = Localizer.new(model.pnet.outnode.children[5])
  self.lsm = nn.LogSoftMax():cuda()
  self.amp = nn.SpatialAdaptiveMaxPooling(cfg.roi_pooling.kw, cfg.roi_pooling.kh):cuda()
end

function Detector:detect(input)
  local cfg = self.model.cfg
  local pnet = self.model.pnet
  local cnet = self.model.cnet
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local bgclass = cfg.class_count + 1   -- background class
  local amp = self.amp
  local lsm = self.lsm
  local cnet_input_planes = self.model.layers[#self.model.layers].filters
  
  local input_size = input:size()
  local input_rect = Rect.new(0, 0, input_size[3], input_size[2])
  
  -- pass image through network
  pnet:evaluate()
  input = input:cuda()
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
                    
          -- classification
          local c = lsm:forward(cls_out)
          --if c[1] > c[2] then
          if math.exp(c[1]) > 0.95 then
            -- regression
            local a = self.anchors:get(i,a,y,x)
            local r = Anchors.anchorToInput(a, reg_out)
            if r:overlaps(input_rect) then
              table.insert(matches, { p=c[1], a=a, r=r, l=i })
            end
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
    
    local iou_threshold = 0.25
    local pick = nms(bb, iou_threshold, score)
    --local pick = nms(bb, iou_threshold, 'area')
    local candidates = {}
    pick:apply(function (x) table.insert(candidates, matches[x]) end )
    
    print(string.format('candidates: %d', #candidates))
    
    -- REGION CLASSIFICATION 
    cnet:evaluate()
    
    -- create cnet input batch
    local cinput = torch.CudaTensor(#candidates, cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
    for i,v in ipairs(candidates) do
      -- pass through adaptive max pooling operation
      local pi, idx = extract_roi_pooling_input(v.r, self.localizer, outputs[5])
      cinput[i] = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
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
      print(x.class)
      if x.class ~= bgclass and math.exp(x.confidence) > 0.2 then
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
   
  return winners
end
