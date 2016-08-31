local image = require 'image'
local cudnn = require 'cudnn'
local cunn = require 'cunn'

require 'Rect'

-- create colors for bounding box drawing
local red = torch.Tensor({1,0,0})
local green = torch.Tensor({0,1,0})
local blue = torch.Tensor({0,0,1})
local white = torch.Tensor({1,1,1})
local colors = { red, green, blue, white }


local width, height = 18,18
local kW_, kH_ = 3,3
local dW_, dH_ = 2,2 --TODO verify
local padW_, padH_ =  0,0 --(kW_-1)/2,(kH_-1)/2
local px1, py1 =6,6
local px2,py2 = 8,8
local in_cp1x = px1
local in_cp2x = px2
local in_cp1y = py1
local in_cp2y = py2
local inpRec1 = Rect.fromCenterWidthHeight(in_cp1x, in_cp1y, 3, 3)
local inpRec2 = Rect.fromCenterWidthHeight(in_cp2x, in_cp2y, 3, 3)
local VERSION = "CENTER2"

--local input = torch.CudaTensor(1,3,width,height)

local function draw_rectangle(img, rect, color)
  local sz = img:size()
  print(rect)
  local x0 = rect.minX
  local x1 = rect.maxX -1
  local y0 = rect.minY
  local y1 = rect.maxY -1

  local w = rect:width()
  if w >= 0 then
    local v = color:view(3,1):expand(3, w)
    img[{{},{}, {x0, x1}, y0}] = v
    img[{{},{}, {x0, x1}, y1}] = v
  end

  local h = rect:height()
  if h >= 0 then
    local v = color:view(3,1):expand(3, h)
    img[{{},{}, x0, {y0, y1}}] = v
    img[{{},{}, x1, {y0, y1}}] = v
  end

  return img
end


local function convLayer_inp2feat(cpx,cpy, kW, kH, dW, dH, padW, padH)
  local cpx_
  local cpy_
  cpx_ = ((cpx + padW - kW/2) / (dW) + 0.5)
  cpy_ = ((cpy + padH - kH/2) / (dH) + 0.5)
  return cpx_,cpy_
end


local function convLayer_inp2feat_inv(cpx,cpy, kW, kH, dW, dH, padW, padH)
  local cpx_
  local cpy_
  cpx_ = ((cpx - 0.5) * (dW) - padW + kW/2) 
  cpy_ = ((cpy - 0.5) * (dH) - padH + kH/2) --math.ceil
  return cpx_,cpy_
end


local function convLayer_inpRec2featRec(rec, kW, kH, dW, dH, padW, padH)
  if VERSION == "CENTER" then
    local cpx, cpy = rec:center()
    local cpx_,cpy_ = convLayer_inp2feat(torch.ceil(cpx),torch.ceil(cpy), kW, kH, dW, dH, padW, padH)
    local owidth_  = torch.floor((rec:width()  + 2 * padW - kW) / dW + 1)
    local oheight_  = torch.floor((rec:height() + 2 * padH - kH) / dH + 1)
    return Rect.fromCenterWidthHeight(cpx_, cpy_, owidth_, oheight_)
  else
    local rec_ =rec:clone()
    rec_.minX = (rec_.minX + padW) / dW
    rec_.minY = (rec_.minY + padH) / dH
    rec_.maxX = (rec_.maxX  + dW - kW + padW) / dW
    rec_.maxY = (rec_.maxY  + dH - kH + padH) / dH
    return rec_
  end
end


local function convLayer_featRec2inpRec(rec, kW, kH, dW, dH, padW, padH)
  if VERSION == "CENTER" then
    local cpx, cpy = rec:center()
    local cpx_,cpy_ = convLayer_inp2feat_inv(torch.ceil(cpx),torch.ceil(cpy), kW, kH, dW, dH, padW, padH)
    local owidth_  = (rec:width()  - 1) * dW - 2 * padW + kW
    local oheight_ = (rec:height() - 1) * dH - 2 * padH + kH
    return Rect.fromCenterWidthHeight(cpx_, cpy_, owidth_, oheight_)
  else
    local rec_ =rec:clone()
    rec_.minX = rec_.minX * dW - padW-- 1*1-1 = 0 --> 0*1-1 = -1 --> -1*1-1 = -2 --> ... --> -74
    rec_.minY = rec_.minY * dH - padH
    rec_.maxX = rec_.maxX * dW - padW + kW - dW
    rec_.maxY = rec_.maxY * dH - padH + kH - dH
    return rec_
  end
end


local function create_layer_info(modules)
  local info = {}
  for i,m in ipairs(modules) do
    if m.kW and m.kH then
      table.insert(info, { kW=m.kW, kH=m.kH, dW=m.dW or 1, dH=m.dH or 1, padW=m.padW or 0, padH=m.padH or 0 })
      print(string.format("Found model information: Index = %d, kW=%d, kH=%d, dW=%d, dH=%d, padW=%d, padH=%d",i,info[#info].kW,info[#info].kH,info[#info].dW,info[#info].dH,info[#info].padW,info[#info].padH))
    end
  end
  return info
end


--- create model structure
local model = nn.Sequential()

local function Block(...)
  local arg = {...}
  model:add(nn.SpatialConvolution(...))
  --model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  --model:add(nn.ReLU(true))
  return model
end


Block(3,1,kW_,kH_,dW_,dH_,padW_, padH_)
--Block(1,1,kW,kH,dW,dH,padW, padH)
print(model)
create_layer_info(model)

for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do
  v.weight:normal(0,0.05)
  v.bias:zero()
end

model:evaluate()
model:cuda()

--- prepare input
local input = torch.CudaTensor(1,3,width,height)
input:zero()
input[1][1][px1][py1] = 100
input[1][1][px2][py2] = 100
local out = model:forward(input:cuda())
local output = torch.repeatTensor(out,1,3,1,1)

local owidth
local oheight

local inpRec1_, inpRec2_
local outRec1, outRec2

--foward
outRec1 = convLayer_inpRec2featRec(inpRec1, kW_, kH_, dW_, dH_, padW_, padH_)
outRec2 = convLayer_inpRec2featRec(inpRec2, kW_, kH_, dW_, dH_, padW_, padH_)

--backward
local outRec1_recovered = convLayer_featRec2inpRec(outRec1,kW_, kH_, dW_, dH_, padW_, padH_)
local outRec2_recovered = convLayer_featRec2inpRec(outRec2, kW_, kH_, dW_, dH_, padW_, padH_)

local is = input:size()
local os = output:size()


owidth  = torch.floor((width    + 2 * padW_ - kW_) / dW_ + 1)
oheight = torch.floor((height   + 2 * padH_ - kH_) / dH_ + 1)
if (dW_*(owidth - 1)+ kW_) < width or (dH_*(oheight - 1)+ kH_) < height then
  print ("The last columns or last lines of the input images will not be taken in account")
end

assert(is[4] == width and is[3] ==  height, string.format("input width = %01f, height = %01f, gt width = %01f, height = %01f",is[3], is[4], width, height))
assert(os[4] ==  owidth and os[3] ==  oheight, string.format("output width = %01f, height = %01f, calculated width = %01f, height = %01f",os[4], os[3], owidth, oheight))

input= draw_rectangle(input, inpRec1, green*100)
input= draw_rectangle(input, inpRec2, green*100)
--image.display(input)
print("inpRec1:center()")
print(inpRec1:center())
print("outRec1:center()")
print(outRec1:center())
print("outRec1_recovered:center()")
print(outRec1_recovered:center())
input= draw_rectangle(input, outRec1_recovered, blue*100)
input= draw_rectangle(input, outRec2_recovered, blue*100)
image.display(input)

--output= draw_rectangle(output, outRec1:snapToInt(), green*20)
--output= draw_rectangle(output, outRec2:snapToInt(), green*20)
--image.display(output)

