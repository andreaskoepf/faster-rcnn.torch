require 'Localizer'

local Anchors = torch.class('Anchors')

function Anchors:__init(proposal_net, scales)
  -- create localizers
  self.localizers = {}
  for i=1,#scales do
    self.localizers[i] = Localizer.new(proposal_net.outnode.children[i])
  end
  
  -- generate vertical and horizontal min-max anchor lookup tables
  local width, height = 200, 200  -- max size of feature layers
  
  -- indicies: scale, aspect-ratio, i, min/max 
  self.w = torch.Tensor(#scales, 3, width, 2)
  self.h = torch.Tensor(#scales, 3, height, 2)
  
  for i,s in ipairs(scales) do
    -- width, height for boxes with s^2 pixels with aspect ratios 1:1, 2:1, 1:2
    local a = s / math.sqrt(2)        
    local aspects = { { s, s }, { 2*a, a }, { a, 2*a } }  
    
    for j,b in ipairs(aspects) do
      local l = self.localizers[i]
      for y=1,height do
        local r = l:featureToInputRect(0, y-1, 0, y)
        local centerX, centerY = r:center()
        r = Rect.fromCenterWidthHeight(centerX, centerY, b[1], b[2])
        self.h[{i, j, y, 1}] = r.minY
        self.h[{i, j, y, 2}] = r.maxY
      end
      
      for x=1,width do
        local r = l:featureToInputRect(x-1, 0, x, 0)
        local centerX, centerY = r:center()
        r = Rect.fromCenterWidthHeight(centerX, centerY, b[1], b[2])
        self.w[{i, j, x, 1}] = r.minX
        self.w[{i, j, x, 2}] = r.maxX
      end
    end
  end
end

function Anchors:get(layer, aspect, y, x)
  local w, h = self.w, self.h
  return Rect.new(w[{layer, aspect, x, 1}], h[{layer, aspect, y, 1}], w[{layer, aspect, x, 2}], h[{layer, aspect, y, 2}])
end

function Anchors:findRangesXY(rect, clip_rect)
  local function lower_bound(t, value)
    local low, high = 1, t:nElement()
    while low <= high do
      local mid = math.floor((low + high) / 2)
      if t[mid] >= value then high = mid - 1
      elseif t[mid] < value then low = mid + 1 end
    end
    return low
  end
  
  local ranges = {}
  local w,h = self.w, self.h
  for i=1,4 do    -- scales
    for j=1,3 do    -- aspect ratios
    
      local clx, cly, cux, cuy  -- lower and upper bounds of clipping rect (indices)
      if clip_rect then
         -- all vertices of anchor must lie in clip_rect (e.g. input image rect)
        clx = lower_bound(w[{i, j, {}, 1}], clip_rect.minX)    -- a.minX >= r.minX
        cly = lower_bound(h[{i, j, {}, 1}], clip_rect.minY)    -- a.minY >= r.minY
        cux = lower_bound(w[{i, j, {}, 2}], clip_rect.maxX)    -- a.maxX <= r.maxX
        cuy = lower_bound(h[{i, j, {}, 2}], clip_rect.maxY)    -- a.maxY <= r.maxY
      end
    
      local l = { layer = i, aspect = j }
      
      -- at least one vertex must lie in rect
      l.lx = lower_bound(w[{i, j, {}, 2}], rect.minX)   -- a.maxX > r.minX 
      l.ly = lower_bound(h[{i, j, {}, 2}], rect.minY)   -- a.maxY > r.minY
      l.ux = lower_bound(w[{i, j, {}, 1}], rect.maxX)   -- a.minX > r.maxX
      l.uy = lower_bound(h[{i, j, {}, 1}], rect.maxY)   -- a.minY > r.maxY
      
      if clip_rect then
        l.lx = math.max(l.lx, clx)
        l.ly = math.max(l.ly, cly)
        l.ux = math.min(l.ux, cux)
        l.uy = math.min(l.uy, cuy)
      end
      
      if l.ux > l.lx and l.uy > l.ly then
        l.xs = w[{i, j, {l.lx, l.ux-1}, {}}]
        l.ys = h[{i, j, {l.ly, l.uy-1}, {}}]
        ranges[#ranges+1] = l
      end      

    end
  end
  
  return ranges
end

function Anchors:findPositive(roi_list, clip_rect, pos_threshold, neg_threshold, include_best)
  local matches = {}
  local best
  
  for i,roi in ipairs(roi_list) do
  
    if include_best then
      best = Rect.empty()   -- best is set to nil if a positive entry was found
    end 
    
    -- evaluate IoU for all overlapping anchors
    local ranges = self:findRangesXY(roi.rect, clip_rect)
    for j,r in ipairs(ranges) do
      -- generate all candidate anchors from xs,ys ranges list
      for y=1,r.ys:size()[1] do
        local minY, maxY = r.ys[{y, 1}], r.ys[{y, 2}]        
        for x=1,r.xs:size()[1] do
          -- create rect, add layer & aspect info
          local anchor_rect = Rect.new(r.xs[{x, 1}], minY, r.xs[{x, 2}], maxY)
          anchor_rect.layer = r.layer
          anchor_rect.aspect = r.aspect 
          anchor_rect.index = { { r.aspect * 6 - 5, r.aspect * 6 }, r.ly + y - 1, r.lx + x - 1 }
          
          local v = Rect.IoU(roi.rect, anchor_rect)
          if v > pos_threshold then
            table.insert(matches, { anchor_rect, roi })
            best = nil
          elseif v > neg_threshold and best and v > Rect.IoU(roi.rect, best) then
            best = anchor_rect
          end
        end
      end
    end

    if best and not best:isEmpty() then
      table.insert(matches, { best, roi })
    end
    
  end
  
  return matches
end

function Anchors:sampleNegative(image_rect, roi_list, neg_threshold, count)
  -- get ranges for all anchors inside image
  local ranges = self:findRangesXY(image_rect, image_rect)
  
  -- use random sampling
  local neg = {}
  local retry = 0
  while #neg < count and retry < 500 do
    
    -- select random anchor
    local r = ranges[torch.random() % #ranges + 1]
    local x = torch.random() % r.xs:size()[1] + 1
    local y = torch.random() % r.ys:size()[1] + 1
    
    local anchor_rect = Rect.new(r.xs[{x, 1}], r.ys[{y, 1}], r.xs[{x, 2}], r.ys[{y, 2}])
    anchor_rect.layer = r.layer
    anchor_rect.aspect = r.aspect 
    anchor_rect.index = { { r.aspect * 6 - 5, r.aspect * 6 }, r.ly + y - 1, r.lx + x - 1 }
   
    -- test against all rois
    local match = false
    for j,roi in ipairs(roi_list) do
      if Rect.IoU(roi.rect, anchor_rect) > neg_threshold then
        match = true
        break
      end 
    end
    
    if not match then
      retry = 0
      table.insert(neg, anchor_rect)
    else
      retry = retry + 1 
    end
    
  end
  
  return neg
end

function Anchors.inputToAnchor(anchor, rect)
  local x = (rect.minX - anchor.minX) / anchor:width()
  local y = (rect.minY - anchor.minY) / anchor:height()
  local w = math.log(rect:width() / anchor:width())
  local h = math.log(rect:height() / anchor:height())
  return torch.FloatTensor({x, y, w, h})
end

function Anchors.anchorToInput(anchor, t)
  return Rect.fromXYWidthHeight(
    t[1] * anchor:width() + anchor.minX,
    t[2] * anchor:height() + anchor.minY,
    math.exp(t[3]) * anchor:width(),
    math.exp(t[4]) * anchor:height() 
  )
end
