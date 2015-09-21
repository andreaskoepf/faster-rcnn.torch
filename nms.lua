-- Non-maximum suppression (NMS)
--
-- Greedily skip boxes that are significantly overlapping a previously 
-- selected box.
--
-- Arguments
--   boxes     Bounding boxes as nx4 tensor, each row specifies the
--             vertices of one box { min_x, min_y, max_x, max_y }. 
--   overlap   Intersection-over-union (IoU) threshold for suppression,
--             all boxes with va alues higher than this threshold will 
--             be suppressed.
--   scores    (optional) Defines in which order boxes are processed.
--             Either the string 'area' or a tensor holding 
--             score-values. Boxes will be processed sorted descending
--             after this value.
--
-- Return value
--   Indices of boxes remaining after non-maximum suppression.

-- Original author: Francisco Massa: https://github.com/fmassa/object-detection.torch 
-- Based on matlab code by Pedro Felzenszwalb https://github.com/rbgirshick/voc-dpm/blob/master/test/nms.m
-- Minor changes by Andreas Köpf, 2015-09-17 
function nms(boxes, overlap, scores)
   local pick = torch.LongTensor()

  if boxes:numel() == 0 then
    return pick
  end

  local x1 = boxes[{{}, 1}]
  local y1 = boxes[{{}, 2}]
  local x2 = boxes[{{}, 3}]
  local y2 = boxes[{{}, 4}]
    
  local area = torch.cmul(x2 - x1 + 1, y2 - y1 + 1)
  
  if type(scores) == 'number' then
    scores = boxes[{{}, scores}]
  elseif scores == 'area' then
    scores = area
  else
    scores = y2   -- use max_y
  end
  
  local v, I = scores:sort(1)

  pick:resize(area:size()):zero()
  local count = 1
  
  local xx1 = boxes.new()
  local yy1 = boxes.new()
  local xx2 = boxes.new()
  local yy2 = boxes.new()

  local w = boxes.new()
  local h = boxes.new()

  while I:numel() > 0 do 
    local last = I:size(1)
    local i = I[last]
    
    pick[count] = i
    count = count + 1
    
    if last == 1 then
      break
    end
    
    I = I[{{1, last-1}}] -- remove picked element from view
    
    -- load values 
    xx1:index(x1, 1, I)
    yy1:index(y1, 1, I)
    xx2:index(x2, 1, I)
    yy2:index(y2, 1, I)
    
    -- compute intersection area
    xx1:cmax(x1[i])
    yy1:cmax(y1[i])
    xx2:cmin(x2[i])
    yy2:cmin(y2[i])
    
    w:resizeAs(xx2)
    h:resizeAs(yy2)
    torch.add(w, xx2, -1, xx1):add(1):cmax(0)
    torch.add(h, yy2, -1, yy1):add(1):cmax(0)
    
    -- reuse existing tensors
    local inter = w:cmul(h)
    local IoU = h
    
    -- IoU := i / (area(a) + area(b) - i)
    xx1:index(area, 1, I) -- load remaining areas into xx1
    torch.cdiv(IoU, inter, xx1 + area[i] - inter) -- store result in iou
    
    I = I[IoU:le(overlap)] -- keep only elements with a IoU < overlap 
  end

  -- reduce size to actual count
  pick = pick[{{1, count-1}}]
  return pick
end
