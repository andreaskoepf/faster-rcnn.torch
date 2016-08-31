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


local function convLayer_inpRec2featRec(rec, kW, kH, dW, dH, padW, padH)
    local rec_ =rec:clone()
    rec_.minX = (rec_.minX + padW) / dW
    rec_.minY = (rec_.minY + padH) / dH
    rec_.maxX = (rec_.maxX  + dW - kW + padW) / dW
    rec_.maxY = (rec_.maxY  + dH - kH + padH) / dH
    return rec_
end


local function convLayer_featRec2inpRec(rec, kW, kH, dW, dH, padW, padH)
    local rec_ =rec:clone()
    rec_.minX = rec_.minX * dW - padW-- 1*1-1 = 0 --> 0*1-1 = -1 --> -1*1-1 = -2 --> ... --> -74
    rec_.minY = rec_.minY * dH - padH
    rec_.maxX = rec_.maxX * dW - padW + kW - dW
    rec_.maxY = rec_.maxY * dH - padH + kH - dH
    return rec_:clone()
end


function Localizer:inputToFeatureRect(rect, layer_index)
  layer_index = layer_index or #self.layers
  for i=1,layer_index do
    local l = self.layers[i]
    rect = convLayer_inpRec2featRec(rect, l.kW, l.kH, l.dW, l.dH, l.padW, l.padH)
  end -- for i=1,layer_index do
  return rect
end


function Localizer:featureToInputRect(minX, minY, maxX, maxY, layer_index)
  layer_index = layer_index or #self.layers
  local rect = Rect.new(minX, minY, maxX, maxY)
  for i=layer_index,1,-1 do
    local l = self.layers[i]
    rect = convLayer_featRec2inpRec(rect, l.kW, l.kH, l.dW, l.dH, l.padW, l.padH)
  end
  return rect
end
