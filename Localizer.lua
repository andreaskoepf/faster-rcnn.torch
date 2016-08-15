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
      end
    end
    return info
  end
  
  self.layers = create_layer_info(trace_modules(outnode))
end

function Localizer:inputToFeatureRect(rect, layer_index)
  layer_index = layer_index or #self.layers
  for i=1,layer_index do
    local l = self.layers[i]
    if l.dW < l.kW then
      rect = rect:inflate((l.kW-l.dW), (l.kH-l.dH))
    end

    rect = rect:offset(l.padW, l.padH)
    
    -- reduce size, keep only filters that fit completely into the rect (valid convolution)
    rect.minX = rect.minX / l.dW
    rect.minY = rect.minY / l.dH
    if (rect.maxX-l.kW) % l.dW == 0 then
      rect.maxX = math.max((rect.maxX-l.kW)/l.dW + 1, rect.minX+1)
    else
      rect.maxX = math.max(math.ceil((rect.maxX-l.kW) / l.dW) + 1, rect.minX+1)
    end
    if (rect.maxY-l.kH) % l.dH == 0 then
      rect.maxY = math.max((rect.maxY-l.kH)/l.dH + 1, rect.minY+1)
    else
      rect.maxY = math.max(math.ceil((rect.maxY-l.kH) / l.dH) + 1, rect.minY+1)
    end

  end
  return rect:snapToInt()
end

function Localizer:featureToInputRect(minX, minY, maxX, maxY, layer_index)
  layer_index = layer_index or #self.layers
  for i=layer_index,1,-1 do
    local l = self.layers[i]
    minX = minX * l.dW - l.padW
    minY = minY * l.dH - l.padH
    maxX = maxX * l.dW - l.padW + l.kW - l.dW
    maxY = maxY * l.dH - l.padH + l.kH - l.dH
  end
  return Rect.new(minX, minY, maxX, maxY)
end

