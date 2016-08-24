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

function Localizer:inputToFeatureRect(rect, inputImg, featureMap, layer_index)
  local inputImg = inputImg or nil
  local featureMap = featureMap or nil
  local width  = rect:width()
  local height = rect:height()
  local centerX, centerY = rect:center()

  if inputImg ~= nil and featureMap ~= nil then
    centerX = featureMap:size(3) * centerX / inputImg:size(3)
    centerY = featureMap:size(2) * centerY / inputImg:size(2)
    width = featureMap:size(3) * width / inputImg:size(3)
    height = featureMap:size(2) * height / inputImg:size(2)
  end

  layer_index = layer_index or #self.layers
  for i=1,layer_index do
    local l = self.layers[i]

    -- Output sizes: (see https://github.com/torch/nn/blob/master/doc/convolution.md)
    -- Conv (i.e. SpatialConvolution) and Pool (i.e. SpatialMaxPooling):
    -- owidth  = floor((width  + 2*padW - kW) / dW + 1)
    -- oheight = floor((height + 2*padH - kH) / dH + 1)

    --width  = math.floor((width  + 2*l.padW - l.kW) / l.dW + 1)
    --height = math.floor((height + 2*l.padH - l.kH) / l.dH + 1)
    -- For the convolution layers 'floor' is correct, but for the maxPooling layers 
    -- we actually have to take 'ceil', here (see model_utilities.lua)...

    -- This calculation seems to be wrong, but is used for backward compatibility
    if inputImg == nil or featureMap == nil then
      
      rect.minX = (rect.minX + l.padW) / l.dW
      rect.maxX = (rect.maxX + l.dW - l.kW + l.padW) / l.dW
      rect.minY = (rect.minY + l.padH) / l.dH
      rect.maxY = (rect.maxY + l.dH - l.kH + l.padH) / l.dH
      
      --[[
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
      ]]
    end -- end of calculation used for backward compatibility

  end -- for i=1,layer_index do

  if inputImg ~= nil and featureMap ~= nil then
    rect = Rect.fromCenterWidthHeight(centerX, centerY, width, height)
  end

  return rect:snapToInt()
end


function Localizer:featureToInputRect(minX, minY, maxX, maxY, inputImg, featureMap, layer_index)
  local inputImg = inputImg or nil
  local featureMap = featureMap or nil
  layer_index = layer_index or #self.layers -- number of layers with a kernel width kW and height kH 
                                            -- (i.e. number of Spatial Convolution and MaxPooling layers)
  local width  = maxX - minX
  local height = maxY - minY

  local centerX = (minX + maxX) / 2
  local centerY = (minY + maxY) / 2
  if inputImg ~= nil and featureMap ~= nil then
    centerX = inputImg:size(3) * centerX / featureMap:size(3)
    centerY = inputImg:size(2) * centerY / featureMap:size(2)
    width = inputImg:size(3) * width / featureMap:size(3)
    height = inputImg:size(2) * height / featureMap:size(2)
  end

  for i=layer_index,1,-1 do
    local l = self.layers[i]

    -- Output sizes: (see https://github.com/torch/nn/blob/master/doc/convolution.md)
    -- Deconv (i.e. SpatialFullConvolution) and Unpool (i.e. SpatialMaxUnpooling):
    -- owidth  = (width  - 1) * dW - 2*padW + kW + adjW
    -- oheight = (height - 1) * dH - 2*padH + kH + adjH

    --width  = (width  - 1) * l.dW - 2*l.padW + l.kW
    --height = (height - 1) * l.dH - 2*l.padH + l.kH

    -- This calculation seems to be wrong, but is used for backward compatibility
    if inputImg == nil or featureMap == nil then
      minX = minX * l.dW - l.padW -- 1*1-1 = 0 --> 0*1-1 = -1 --> -1*1-1 = -2 --> ... --> -74
      minY = minY * l.dH - l.padH
      maxX = maxX * l.dW - l.padW + l.kW - l.dW
      maxY = maxY * l.dH - l.padH + l.kH - l.dH
    end -- end of calculation used for backward compatibility
  end

  local rect
  if inputImg ~= nil and featureMap ~= nil then
    rect = Rect.fromCenterWidthHeight(centerX, centerY, width, height)
  else
    -- This calculation seems to be wrong, but is used for backward compatibility
    rect = Rect.new(minX, minY, maxX, maxY)
  end
  return rect --:snapToInt()
end
