require 'image'
require 'Anchors'

function load_image(fn, w, h, normalize)
  local img = image.load(path.join(base_path, fn), 3, 'float')
  local originalSize = img:size()
  img = image.rgb2yuv(image.scale(img, w, h))
  if normalize then
    img[1] = normalization:forward(img[{{1}}])
  end
  local scaleX, scaleY = w / originalSize[3], h / originalSize[2]
  return img, scaleX, scaleY
end

function load_image_auto_size(fn, target_smaller_side, max_pixel_size, color_space)
  local img = image.load(path.join(base_path, fn), 3, 'float')
  local dim = img:size()
  
  local w, h
  if dim[2] < dim[3] then
    -- height is smaller than width, set h to target_size
    w = math.min(dim[3] * target_smaller_side/dim[2], max_pixel_size)
    h = dim[2] * w/dim[3]
  else
    -- width is smaller than height, set w to target_size
    h = math.min(dim[2] * target_smaller_side/dim[1], max_pixel_size)
    w = dim[3] * h/dim[2]
  end
  
  img = image.scale(img, w, h)
  
  if color_space == 'yuv' then
    img = image.rgb2yuv(img)
  elseif color_space == 'lab' then
    img = image.rgb2lab(img)
  elseif color_space == 'hsv' then
    img = image.rgb2hsv(img)
  end

  return img, dim
end

local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(11))

function normalize_image(img)
  img[1] = normalization:forward(img[{{1}}])
  return img
end


local BatchIterator = torch.class('BatchIterator')

local function randomize_order(...)
  local sets = { ... }
  for i,x in ipairs(stes) do
    if #x.list > 0 then   -- e.g. background examples are optional and randperm does not like 0 count
      x.order:randperm(#x.list)   -- shuffle
    end
    x.i = 0   -- reset index positions (will be incremented first)
  end
end

local function next_entry(set)
  if set.i >= #set.list then
    randomize_order(set)
  end
  set.i = set.i + 1
  return set.list[set.order[set.i]]
end


function BatchIterator:__init(pnet, training_data)
  self.ground_truth = training_data.ground_truth 
  self.anchors = Anchors.new(pnet, training_data.scales)
  
  -- index tensors define evaluation order
  self.training = { order = torch.IntTensor(), i = 0, list = training_data.train_file_names }
  self.background = { order = torch.IntTensor(), i = 0, list = training_data.background_file_names }
  self.test = { order = torch.IntTensor(), i = 0, list = training_data.test_file_names }
  
  randomize_order(self.training, self.background, self.test)
end


--[[
 Load image data
]]
function BatchIterator:nextTraining(count)
  local batch = {}
  while count > 0 do
  
    if self.training_order > self.training_order:size()[1] then
      self.randomize()
    end
    
    x.scaleX = img:size()[3] / original_size[3]
    x.scaleY = img:size()[2] / original_size[2] 
  
    -- select random image
    local fn = next_entry(self.training)
    local rois = self.ground_truth[fn].rois
         
          -- load image
    --print(fn)
    local img = load_image_auto_size(fn, training_data.target_smaller_side, training_data.max_pixel_size, 'yuv')
    local img_size = img:size()
    img = normalize_image(img)
  
  end
  
  return batch
end


function BatchIterator:nextTesting(count)

end