require 'Rect'
require 'utilities'

local Localizer = torch.class('Localizer')

function Localizer:__init(outnode)

  local function trace_modules(node)
    local modules = {}
    local function add_modules(c)
      if c.modules then
        for i=#c.modules,1,-1 do
          add_modules(c.modules[i])
        end
      else
        table.insert(modules, c)
      end
    end
    while node do
      if node.data.module then
        add_modules(node.data.module)
      end
      node = node.children and node.children[1]
    end
    return reverse(modules)
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

  self.layers = create_layer_info(trace_modules(outnode))
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

  local cpx, cpy = rec:center()
  local cpx_,cpy_ = convLayer_inp2feat(cpx,cpy, kW, kH, dW, dH, padW, padH)
  local owidth_  = ((rec:width()  + 2 * padW - kW) / dW + 1)
  local oheight_  = ((rec:height() + 2 * padH - kH) / dH + 1)
  return Rect.fromCenterWidthHeight(cpx_, cpy_, owidth_, oheight_)
  
--[[
  rec.minX = rec.minX * dW - padW -1 -- 1*1-1 = 0 --> 0*1-1 = -1 --> -1*1-1 = -2 --> ... --> -74
  rec.minY = rec.minY * dH - padH -1
  rec.maxX = rec.maxX * dW - padW + kW - dW -1
  rec.maxY = rec.maxY * dH - padH + kH - dH -1
  return rec
  --]]
end


local function convLayer_inpRec2featRec_inv(rec, kW, kH, dW, dH, padW, padH)
  
  local cpx, cpy = rec:center()
  local cpx_,cpy_ = convLayer_inp2feat_inv(cpx,cpy, kW, kH, dW, dH, padW, padH)
  local owidth_  = (rec:width()  - 1) * dW - 2*padW + kW --rec:width() + dW - kW
  local oheight_ = (rec:height() - 1) * dH - 2*padH + kH
  return Rect.fromCenterWidthHeight(cpx_, cpy_, owidth_, oheight_)

--[[
  rec.minX = (rec.minX + 1 + padW)/dW 
  rec.minY = (rec.minY + 1 + padH)/dH
  rec.maxX = (rec.maxX +1 +dW - kW +padW)/dW
  rec.maxY = (rec.maxY +1 +dH - kH +padH)/dH
  return rec
  --]]
end

function Localizer:inputToFeatureRect(rect, layer_index)
  layer_index = layer_index or #self.layers
  for i=1,layer_index do
    local l = self.layers[i]
    rect = convLayer_inpRec2featRec(rect, l.kW, l.kH, l.dW, l.dH, l.padW, l.padH)
  end -- for i=1,layer_index do
  return rect:snapToInt()
end


function Localizer:featureToInputRect(minX, minY, maxX, maxY, layer_index)
  layer_index = layer_index or #self.layers
  local rect = Rect.new(minX, minY, maxX, maxY)
  for i=layer_index,1,-1 do
    local l = self.layers[i]
    rect = convLayer_inpRec2featRec_inv(rect, l.kW, l.kH, l.dW, l.dH, l.padW, l.padH)
  end
  return rect:snapToInt()
end
