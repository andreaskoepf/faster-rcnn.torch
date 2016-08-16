require 'Rect'
require 'utilities'
--[[ using scales 8 16 32
array([[ -83.,  -39.,  100.,   56.],
       [-175.,  -87.,  192.,  104.],
       [-359., -183.,  376.,  200.],
       [ -55.,  -55.,   72.,   72.],
       [-119., -119.,  136.,  136.],
       [-247., -247.,  264.,  264.],
       [ -35.,  -79.,   52.,   96.],
       [ -79., -167.,   96.,  184.],
       [-167., -343.,  184.,  360.]]
--]]



---Return width, height, x center, and y center for an anchor (window).
function _whctrs(anchor)
  local w ,h , x_ctr, y_ctr
  local anchor_
  if torch.type(anchor) == 'Rect' then
    anchor_ = anchor:totensor()
  else
    anchor_ = anchor
  end
  w = anchor_[3] - anchor_[1] + 1
  h = anchor_[4] - anchor_[2] + 1
  x_ctr = anchor_[1] +  (w - 1) * 0.5
  y_ctr = anchor_[2] +  (h - 1) * 0.5
  return w, h, x_ctr, y_ctr
end

---Given a vector of widths (ws) and heights (hs) around a center (x_ctr, y_ctr), output a set of anchors (windows).
function mkanchors(ws, hs, x_ctr, y_ctr, isCenter)
  local isCenter = isCenter  or false
  local cx ,cy,wx,wy
  if not isCenter then
    return Rect.fromXYWidthHeight(x_ctr, y_ctr, ws, hs)
  else
    cx = x_ctr
    cy = y_ctr
    wx = ws-1
    wy = hs-1
  end
  return Rect.fromCenterWidthHeight(cx, cy, wx, wy)
end

---Enumerate a set of anchors for each aspect ratio wrt an anchor.
function _ratio_enum(anchor, ratios)
  local w, h, x_ctr, y_ctr = _whctrs(anchor)
  local size = w * h

  local size_ratios = torch.mul(torch.cinv(ratios),size)

  local ws = torch.round(torch.sqrt(size_ratios))
  local hs = torch.round(torch.cmul(ratios,ws))
  local anchors = {}
  print(hs)
  for i =1, ratios:size()[2] do
    local w =ws[{1,i}]
    local h =hs[{1,i}]
    local x =x_ctr[1]
    local y =y_ctr[1]
    local anchor = mkanchors(w, h, x, y,true)
    table.insert(anchors,anchor)
  end
  return anchors
end

---Enumerate a set of anchors for each scale wrt an anchor.
function _scale_enum(anchor, scales)
  local anchors = {}
  local anchor_
  local w, h, x_ctr, y_ctr = _whctrs(anchor)

  for i = 1,scales:size()[2] do
    print(scales[{1,i}])
    local ws = scales[{1,i}]*w
    local hs = scales[{1,i}]*h
    table.insert(anchors, mkanchors(ws, hs, x_ctr, y_ctr,true))
  end
  return anchors
end

---Generate anchor (reference) windows by enumerating aspect ratios X scales wrt a reference (0, 0, 15, 15) window.
function generate_anchors(scales, base_size, ratios)
  scales = scales or torch.Tensor(3, 6)
  base_size = base_size or 16
  ratios= ratios or torch.Tensor({{0.5, 1, 2}})

  local base_anchor = torch.Tensor({{1}, {1}, {base_size}, {base_size}}) - 1
  print(base_anchor)
  local ratio_anchors = _ratio_enum(base_anchor, ratios)
  local anchors = {}
  for i = 1, #ratio_anchors do
    table.insert(anchors, _scale_enum(ratio_anchors[i], scales))
  end
  return anchors
end

--[[
--local scales = torch.Tensor({{8, 16, 32}})
local scales = torch.Tensor({{8}})
myscales = generate_anchors(scales)

for i = 1,#myscales do
  for j = 1,#myscales[i] do
    print(myscales[i][j])
  end
end
print(myscales)
--]]