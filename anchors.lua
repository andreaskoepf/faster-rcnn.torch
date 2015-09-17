
function generate_anchors(layer_sizes, input_width, input_height, noCrossBorder)
  local anchors = {}
  local inputRect = Rect.new(0, 0, input_width, input_height)

  --local scales = { 32, 64, 128, 256 }
  local scales = { 32, 64, 128, 256 }
  local factors = { 8, 16, 16, 16 }
  for i,l in ipairs(layer_sizes) do
    local h, w = l[2], l[3]
    local borderX = input_width - w * factors[i]
    local borderY = input_height - h * factors[i]
    
    for y=1,h do
      local centerY = (y - 0.5) / h * (input_height - borderY) + 0.5 * borderY
      for x=1,w do
        local centerX = (x - 0.5) / w * (input_width - borderX) + 0.5 * borderX
  
        local s = scales[i]
        
        -- width, height for boxes with s^2 pixels with aspect ratios 1:1, 2:1, 1:2
        local a = s / math.sqrt(2)        
        local boxes = { { s, s }, { 2*a, a }, { a, 2*a } }  
        
        -- generate anchor rects
        for j,sz in ipairs(boxes) do
          local r = Rect.fromCenterWidthHeight(centerX, centerY, sz[1], sz[2])
          if not noCrossBorder or inputRect:contains(r) then
            table.insert(anchors, r)
            r.layer = i
            r.aspect = j
            r.index = { { j * 6 - 5, j * 6 }, y, x }
          end
        end
        
      end -- for over x
    end -- for over y
    
  end -- for over layer_sizes

  return anchors
end

function sample_negative_anchors(roi_list, anchors, neg_threshold, count)
  local neg = {}
  while #neg < count do
    
    -- select random anchor
    local i = torch.random() % #anchors + 1
    local y = anchors[i] 
    local match = false
    
    for j,x in ipairs(roi_list) do
      if Rect.IoU(x.rect, y) > neg_threshold then
        match = true
        break
      end 
    end
    
    if not match then
      table.insert(neg, y)
    end
    
  end
  
  return neg
end

function find_positive_anchors(roi_list, anchors, pos_threshold, neg_threshold, include_best)
  local pos = {}
  local best
  
  for j,x in ipairs(roi_list) do

    if include_best then
      best = Rect.empty()   -- best is set to nil if a positive entry was found
    end 
    
    -- loop over all anchors
    for i,y in ipairs(anchors) do
      local r = Rect.IoU(x.rect, y)
      
      if r > pos_threshold then
        table.insert(pos, { y, x })
        best = nil
      elseif r > neg_threshold and best and r > Rect.IoU(x.rect, best) then
        best = y
      end
    end
    
    if best and not best:isEmpty() then
      table.insert(pos, { best, x })
    end
  end
  
  return pos
end

function input_to_anchor(anchor, rect)
  local x = (rect.minx - anchor.minx) / anchor:width()
  local y = (rect.miny - anchor.miny) / anchor:height()
  local w = math.log(rect:width() / anchor:width())
  local h = math.log(rect:height() / anchor:height())
  return torch.FloatTensor({x, y, w, h})
end

function anchor_to_input(anchor, t)
  return Rect.fromXYWidthHeight(
    t[1] * anchor:width() + anchor.minx,
    t[2] * anchor:height() + anchor.miny,
    math.exp(t[3]) * anchor:width(),
    math.exp(t[4]) * anchor:height() 
  )
end