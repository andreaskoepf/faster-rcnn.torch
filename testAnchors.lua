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


local width, height = 64,64
local kW, kH = 3,3
local dW, dH = 2,2 --TODO verify
local padW, padH = (kW-1)/2,(kH-1)/2
local px1, py1 =20,30
local px2,py2 = 30,18
local in_cp1x = px1
local in_cp2x = px2
local in_cp1y = py1
local in_cp2y = py2
local inpRec1 = Rect.fromCenterWidthHeight(in_cp1x, in_cp1y, 3, 3)
local inpRec2 = Rect.fromCenterWidthHeight(in_cp2x, in_cp2y, 3, 3)

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
   cpx_  = ((cpx - 0.5) * (dW) - padW + kW/2)
   cpy_  = ((cpy - 0.5) * (dH) - padH + kH/2)  --math.ceil
  return cpx_,cpy_
end


local function convLayer_inpRec2featRec(rec, kW, kH, dW, dH, padW, padH)
--[[
  local cpx, cpy = rec:center()
  local cpx_,cpy_ = convLayer_inp2feat(cpx,cpy, kW, kH, dW, dH, padW, padH)
  local owidth_  = ((rec:width()  + 2 * padW - kW) / dW + 1)
  local oheight_  = ((rec:height() + 2 * padH - kH) / dH + 1)
  return Rect.fromCenterWidthHeight(cpx_, cpy_, owidth_, oheight_)
  --]]

  rec.minX = (rec.minX + 1 + padW)/dW 
  rec.minY = (rec.minY + 1 + padH)/dH
  rec.maxX = (rec.maxX +1 +dW - kW +padW)/dW
  rec.maxY = (rec.maxY +1 +dH - kH +padH)/dH
  return rec
  
end


local function convLayer_featRec2inpRec(rec, kW, kH, dW, dH, padW, padH)

  rec.minX = rec.minX * dW - padW -1-- 1*1-1 = 0 --> 0*1-1 = -1 --> -1*1-1 = -2 --> ... --> -74
  rec.minY = rec.minY * dH - padH -1
  rec.maxX = rec.maxX * dW - padW + kW - dW -1
  rec.maxY = rec.maxY * dH - padH + kH - dH -1
  return rec

--[[
  local cpx, cpy = rec:center()
  local cpx_,cpy_ = convLayer_inp2feat_inv(cpx,cpy, kW, kH, dW, dH, padW, padH)
  local owidth_  = (rec:width()  - 1) * dW - 2*padW + kW --rec:width() + dW - kW
  local oheight_ = (rec:height() - 1) * dH - 2*padH + kH
  return Rect.fromCenterWidthHeight(cpx_, cpy_, owidth_, oheight_)
--]]
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


Block(3,1,kW,kH,dW,dH,padW, padH)
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

local owidth_
local oheight_
local owidth
local oheight

local out_cp1x_, out_cp1y_ = convLayer_inp2feat(in_cp1x,in_cp1y, kW, kH, dW, dH, padW, padH)
--local out_cp2x_, out_cp2y_ = convLayer_inp2feat(out_cp1x_,out_cp1y_, kW, kH, dW, dH, padW, padH)

local out_cp1x, out_cp1y
local out_cp2x, out_cp2y
local inpRec1_, inpRec2_
local outRec1_, outRec2_


out_cp1x, out_cp1y = convLayer_inp2feat(out_cp1x_, out_cp1y_, kW, kH, dW, dH, padW, padH, padW, padH)
out_cp2x, out_cp2y = convLayer_inp2feat(out_cp1x, out_cp1y, kW, kH, dW, dH, padW, padH, padW, padH)
local a, b = convLayer_inp2feat_inv(out_cp1x,out_cp1y, kW, kH, dW, dH, padW, padH)


--assert(math.abs(a - out_cp2x_) <= 1, string.format("Test failed: a = %04f , out_cpx = %04f", a, out_cp2x_))
--assert(math.abs(b - out_cp2y_) <= 1, string.format("Test failed: b = %04f , out_cpy = %04f", b, out_cp2y_))


--owidth_  = math.floor((width    + 2 * padW - kW) / dW + 1)
--oheight_ = math.floor((height   + 2 * padH - kH) / dH + 1)
--owidth   = math.floor((owidth_  + 2 * padW - kW) / dW + 1)
--oheight  = math.floor((oheight_ + 2 * padH - kH) / dH + 1)


inpRec1_ = convLayer_inpRec2featRec(inpRec1, kW, kH, dW, dH, padW, padH)
inpRec2_ = convLayer_inpRec2featRec(inpRec2, kW, kH, dW, dH, padW, padH)
outRec1_ = convLayer_inpRec2featRec(inpRec1_, kW, kH, dW, dH, padW, padH)
outRec2_ = convLayer_inpRec2featRec(inpRec2_, kW, kH, dW, dH, padW, padH)

local outRec1_recovered_ = convLayer_featRec2inpRec(inpRec1_, kW, kH, dW, dH, padW, padH)
local outRec2_recovered_ = convLayer_featRec2inpRec(inpRec2_, kW, kH, dW, dH, padW, padH)
local outRec1_recovered = convLayer_featRec2inpRec(outRec1_recovered_, kW, kH, dW, dH, padW, padH)
local outRec2_recovered = convLayer_featRec2inpRec(outRec2_recovered_, kW, kH, dW, dH, padW, padH)

--print(inpRec1_)
--print(outRec1_recovered)
--os.exit()
print(string.format("feature point index = [%d,%d]",out_cp1x,out_cp1y))
--(string.format("feature point index = [%d,%d]",out_cp2x,out_cp2y))


output[1][1][out_cp1x][out_cp1y] = 20
--output[1][1][out_cp2x][out_cp2y] = 20

local is = input:size()
local os = output:size()

assert(is[3] == math.floor(width) and is[4] ==  math.floor(height), string.format("input width = %01f, height = %01f, gt width = %01f, height = %01f",is[3], is[4], math.floor(width), math.floor(height)))
--assert(os[3] ==  owidth and os[4] ==  oheight, string.format("output width = %01f, height = %01f, gt width = %01f, height = %01f",os[3], os[4], owidth, oheight))

input= draw_rectangle(input, inpRec1:snapToInt(), green*100)
input= draw_rectangle(input, inpRec2:snapToInt(), green*100)
--image.display(input)
input= draw_rectangle(input, outRec1_recovered:snapToInt(), blue*100)
input= draw_rectangle(input, outRec2_recovered:snapToInt(), blue*100)
image.display(input)

--output= draw_rectangle(output, outRec1_, green*20)
--output= draw_rectangle(output, outRec2_, green*20)
--image.display(output)

