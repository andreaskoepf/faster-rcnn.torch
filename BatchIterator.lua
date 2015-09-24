require 'image'
require 'utilities'
require 'Anchors'

local BatchIterator = torch.class('BatchIterator')

local function randomize_order(...)
  local sets = { ... }
  for i,x in ipairs(sets) do
    if x.list and #x.list > 0 then   -- e.g. background examples are optional and randperm does not like 0 count
      x.order:randperm(#x.list)   -- shuffle
    end
    x.i = 1   -- reset index positions
  end
end

local function next_entry(set)
  if set.i > #set.list then
    randomize_order(set)
  end
  
  local fn = set.list[set.order[set.i]]
  set.i = set.i + 1
  return fn
end

local function transform_example(img, rois, fimg, froi)
  local result = {}
  local d = img:size()
  assert(d:size() == 3)
  img = fimg(img, d[3], d[2])   -- transform image
  local dn = img:size()
  local img_rect = Rect.new(0, 0, dn[3], dn[2])
  if rois then
    for i=1,#rois do
      local roi = rois[i]
      roi.rect = froi(roi.rect, d[3], d[2])   -- transform roi
      if roi.rect then
        roi.rect = roi.rect:clip(img_rect) 
        if not roi.rect:isEmpty() then
          result[#result+1] = roi
        end
      end 
    end
  end
  return img, result
end

local function scale(img, rois, scaleX, scaleY)
  scaleY = scaleY or scaleX
  return transform_example(img, rois, 
    function(img, w, h) return image.scale(img, w * scaleX, h * scaleY) end,
    function(r, w, h) return r:scale(scaleX, scaleY) end
  )
end

local function hflip(img, rois)
  return transform_example(img, rois,
    function(img, w, h) return image.hflip(img) end,
    function(r, w, h) return Rect.new(w - r.maxX, r.minY, w - r.minX, r.maxY) end  
  )
end

local function vflip(img, rois)
  return transform_example(img, rois,
    function(img, w, h) return image.vflip(img) end,
    function(r, w, h) return Rect.new(r.minX, h - r.maxY, r.maxX, h - r.minY) end
  )
end

local function crop(img, rois, rect)
  return transform_example(img, rois,
    function(img, w, h) return image.crop(img, rect.minX, rect.minY, rect.maxX, rect.maxY) end,
    function(r, w, h) return r:clip(rect):offset(-rect.minX, -rect.minY) end 
  )
end

function BatchIterator:__init(pnet, training_data)
  local cfg = training_data.cfg
  
  -- bounding box data (defined in pixels on original image)
  self.ground_truth = training_data.ground_truth 
  self.cfg = cfg
  
  if cfg.normalization.method == 'contrastive' then
    self.normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(cfg.normalization.width))
  else
    self.normalization = nn.Identity()
  end
  
  self.anchors = Anchors.new(pnet, cfg.scales)
  
  -- index tensors define evaluation order
  self.training = { order = torch.IntTensor(), list = training_data.train_file_names }
  self.test = { order = torch.IntTensor(), list = training_data.test_file_names }
  self.background = { order = torch.IntTensor(), list = training_data.background_file_names }
  
  randomize_order(self.training, self.test, self.background)
end
  
function BatchIterator:processImage(img, rois)
  local cfg = self.cfg
  
  -- determine optimal resize
  local img_size = img:size()
  local tw, th = find_target_size(img_size[3], img_size[2], cfg.target_smaller_side, cfg.max_pixel_size)
  
  -- random scaling
  local scale_X = tw / img_size[3] + math.random() * 0.5 - 0.25  -- +/- 25% of 'optimal size'
  local scale_Y = scale_X + math.random() * 0.1 - 0.05    -- 5% jitter
  img, rois = scale(img, rois, scale_X, scale_Y)
  
  -- crop image to final size if necesssary
  img_size = img:size()
  if img_size[3] > tw or img_size[2] > th then
    tw, th = math.min(tw, img_size[3]), math.min(th, img_size[2])
    local crop_rect = Rect.fromXYWidthHeight(
      math.floor(math.random() * (img_size[3]-tw)), math.floor(math.random() * (img_size[2]-th)), 
      tw, th
    )
    img, rois = crop(img, rois, crop_rect)
  end
  
  -- perform random flip operatons, in 10% of cases hflip+vflip
  local r = math.random() 
  if r > 0.5 or r > 0.9 then
    img, rois = hflip(img, rois)
  end
  if r > 0.7 then
    img, rois = vflip(img, rois)
  end
  
  img[1] = self.normalization:forward(img[{{1}}])   -- normalize luminance channel img
  
  return img, rois
end
  
function BatchIterator:nextTraining(count)
  local cfg = self.cfg
  local batch = {}
  count = count or cfg.batch_size
  
  -- add a background examples
  if #self.background.list > 0 then
    local fn = next_entry(self.background)
    local img = load_image(fn, cfg.color_space, cfg.background_base_path)
    img = self:processImage(img)
    local img_size = img:size()        -- get final size
    assert(img_size[2] >= 64 and img_size[3] >= 64)
    local img_rect = Rect.new(0, 0, img_size[3], img_size[2])
    local negative = self.anchors:sampleNegative(img_rect, {}, 0, math.floor(count * 0.05))   -- add 5% negative samples per batch
    table.insert(batch, { img = img, positive = {}, negative = negative })
    count = count - #negative
    --print(string.format('background: %s (%dx%d)', fn, img_size[3], img_size[2]))
  end
  
  while count > 0 do
  
    -- select next image file name and get ROIs 
    local fn = next_entry(self.training)
    local rois = deep_copy(self.ground_truth[fn].rois)   -- copy RoIs ground-truth data (will be manipulated)
    
    -- load image
    local img = load_image(fn, cfg.color_space, cfg.examples_base_path)    
    img, rois = self:processImage(img, rois)
    
    local img_size = img:size()        -- get final size
    assert(img_size[2] >= 64 and img_size[3] >= 64)
    
    -- find positive examples
    local img_rect = Rect.new(0, 0, img_size[3], img_size[2])
    local positive = self.anchors:findPositive(rois, img_rect, cfg.positive_threshold, cfg.negative_threshold, cfg.best_match)
    local negative = self.anchors:sampleNegative(img_rect, rois, cfg.negative_threshold, math.max(16, #positive))
    
    -- debug boxes
    --[[
    local red = torch.Tensor({1,0,0})
    for i=1,#negative do
      draw_rectangle(img, negative[i], red)
    end
    local green = torch.Tensor({0,1,0})
    for i=1,#positive do
      draw_rectangle(img, positive[i][1], green)
    end
    image.saveJPG(string.format('test%d.jpg', self.training.i), img)
    ]]
    
    table.insert(batch, { img = img, positive = positive, negative = negative })
    --print(string.format("'%s' (%dx%d); p: %d; n: %d", fn, img_size[3], img_size[2], #positive, #negative))
    count = count - #positive - #negative
  end
  
  return batch
end


function BatchIterator:nextTesting(count)
  -- TODO
end
