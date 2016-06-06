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
    function(img, w, h) return image.scale(img, math.max(1, w * scaleX), math.max(1, h * scaleY)) end,
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

function BatchIterator:__init(model, training_data)
  local cfg = model.cfg

  -- bounding box data (defined in pixels on original image)
  self.ground_truth = training_data.ground_truth
  self.cfg = cfg

  if cfg.normalization.method == 'contrastive' then
    self.normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(cfg.normalization.width))
  else
    self.normalization = nn.Identity()
  end

  self.anchors = Anchors.new(model.pnet, cfg.scales)

  -- index tensors define evaluation order
  self.training = { order = torch.IntTensor(), list = training_data.training_set }
  self.validation = { order = torch.IntTensor(), list = training_data.validation_set }
  self.background = { order = torch.IntTensor(), list = training_data.background_files or {} }

  randomize_order(self.training, self.validation, self.background)
end

function BatchIterator:processImage(img, rois)
  local cfg = self.cfg
  local aug = cfg.augmentation

  -- determine optimal resize
  local img_size = img:size()
  local tw, th = find_target_size(img_size[3], img_size[2], cfg.target_smaller_side, cfg.max_pixel_size)

  local scale_X, scale_Y = tw / img_size[3], th / img_size[2]

  -- random scaling
  if aug.random_scaling and aug.random_scaling > 0 then
    scale_X = tw * (math.random() - 0.5) * aug.random_scaling / img_size[3]
    scale_Y = scale_X + (math.random() - 0.5) * aug.aspect_jitter
  end

  img, rois = scale(img, rois, scale_X, scale_Y)

  -- crop image to final size if we upsampled at least one dimension
  img_size = img:size()
  if img_size[3] > tw or img_size[2] > th then
    tw, th = math.min(tw, img_size[3]), math.min(th, img_size[2])
    local crop_rect = Rect.fromXYWidthHeight(
      math.floor(math.random() * (img_size[3]-tw)),
      math.floor(math.random() * (img_size[2]-th)),
      tw,
      th
    )
    img, rois = crop(img, rois, crop_rect)
  end

  -- horizontal flip operation
  if aug.hflip and aug.hflip > 0 then
    if math.random() < aug.hflip then
      img, rois = hflip(img, rois)
    end
  end

  -- vertical flip operation
  if aug.vflip and aug.vflip > 0 then
    if math.random() < aug.vflip then
      img, rois = vflip(img, rois)
    end
  end

  if cfg.normalization.centering then
    for i = 1,3 do
      img[i] = img[i]:add(-img[i]:mean())
    end
  end

  if cfg.normalization.scaling then
    for i = 1,3 do
      local s = img[i]:std()
      if s > 1e-8 then
        img[i] = img[i]:div(s)
      end
    end
  end

  img[1] = self.normalization:forward(img[{{1}}])   -- normalize luminance channel img

  return img, rois
end

function BatchIterator:nextTraining(count)
  local cfg = self.cfg
  local batch = {}
  count = count or cfg.batch_size

  -- use local function to allow early exits in case of to image load failures
  local function try_add_next()
    local fn = next_entry(self.training)
    local rois = deep_copy(self.ground_truth[fn].rois)   -- copy RoIs ground-truth data (will be manipulated)

    -- load image, wrap with pcall since image net contains invalid non-jpeg files
    local status, img = pcall(function () return load_image(fn, cfg.color_space, cfg.examples_base_path) end)
    if not status then
      -- pcall failed, corrupted image file?
      print(string.format("Invalid image '%s': %s", fn, img))
      return 0
    end

    local img_size = img:size()
    if img:nDimension() ~= 3 or img_size[1] ~= 3 then
      print(string.format("Warning: Skipping image '%s'. Unexpected channel count: %d (dim: %d)", fn, img_size[1], img:nDimension()))
      return 0
    end

    local img, rois = self:processImage(img, rois)
    img_size = img:size()        -- get final size
    if img_size[2] < 128 or img_size[3] < 128 then
      -- notify user about skipped image
      print(string.format("Warning: Skipping image '%s'. Invalid size after process: (%dx%d)", fn, img_size[3], img_size[2]))
      return 0
    end

    -- find positive examples
    local img_rect = Rect.new(0, 0, img_size[3], img_size[2])
    local positive = self.anchors:findPositive(rois, img_rect, cfg.positive_threshold, cfg.negative_threshold, cfg.best_match)

    -- random negative examples
    local negative = self.anchors:sampleNegative(img_rect, rois, cfg.negative_threshold, math.max(1, #positive)) --20160306 16
    local count = #positive + #negative

    if cfg.nearby_aversion then
      local nearby_negative = {}
      -- add all nearby negative anchors
      for i,p in ipairs(positive) do
        local cx, cy = p[1]:center()
        local nearbyAnchors = self.anchors:findNearby(cx, cy)
        for i,a in ipairs(nearbyAnchors) do
          if Rect.IoU(p[1], a) < cfg.negative_threshold then
            table.insert(nearby_negative, { a })
          end
        end
      end

      local c = math.min(#positive, count)
      shuffle_n(nearby_negative, c)
      for i=1,c do
        table.insert(negative, nearby_negative[i])
        count = count + 1
      end
    end

    -- debug boxes
    if false then
      local dimg = img
      if color_space == 'yuv' then
        dimg = image.yuv2rgb(img)
      elseif color_space == 'lab' then
        dimg = image.lab2rgb(img)
      elseif color_space == 'hsv' then
        dimg = image.hsv2rgb(img)
      end

      local red = torch.Tensor({1,0,0})
      local white = torch.Tensor({1,1,1})

      for i=1,#negative do
        draw_rectangle(dimg, negative[i][1], red)
      end
      local green = torch.Tensor({0,1,0})
      for i=1,#positive do
        draw_rectangle(dimg, positive[i][1], green)
      end

      for i=1,#rois do
        draw_rectangle(dimg, rois[i].rect, white)
      end
      image.saveJPG(string.format('anchors%d.jpg', self.training.i), dimg)
    end

    table.insert(batch, { img = img, positive = positive, negative = negative })
    print(string.format("'%s' (%dx%d); p: %d; n: %d", fn, img_size[3], img_size[2], #positive, #negative))
    return count
  end

  -- add a background examples
  if #self.background.list > 0 then
    local fn = next_entry(self.background)
    local status, img = pcall(function () return load_image(fn, cfg.color_space, cfg.background_base_path) end)
    if status then
      img = self:processImage(img)
      local img_size = img:size()        -- get final size
      if img_size[2] >= 128 and img_size[3] >= 128 then
        local img_rect = Rect.new(0, 0, img_size[3], img_size[2])
        local negative = self.anchors:sampleNegative(img_rect, {}, 0, math.floor(count * 0.05))   -- add 5% negative samples per batch
        table.insert(batch, { img = img, positive = {}, negative = negative })
        count = count - #negative
        print(string.format('background: %s (%dx%d)', fn, img_size[3], img_size[2]))
      end
    else
      -- pcall failed, corrupted image file?
      print(string.format("Invalid image '%s': %s", fn, img))
    end
  end

  while count > 0 do
    count = count - try_add_next()
  end

  return batch
end

function BatchIterator:nextValidation(count)
  local cfg = self.cfg
  local batch = {}
  count = count or 1

  -- use local function to allow early exits in case of to image load failures
  while count > 0 do
    local fn = next_entry(self.validation)

    -- load image, wrap with pcall since image net contains invalid non-jpeg files
    local status, img = pcall(function () return load_image(fn, cfg.color_space, cfg.examples_base_path) end)
    if not status then
      -- pcall failed, corrupted image file?
      print(string.format("Invalid image '%s': %s", fn, img))
      goto continue
    end

    local img_size = img:size()
    if img:nDimension() ~= 3 or img_size[1] ~= 3 then
      print(string.format("Warning: Skipping image '%s'. Unexpected channel count: %d (dim: %d)", fn, img_size[1], img:nDimension()))
      goto continue
    end

    local rois = deep_copy(self.ground_truth[fn].rois)   -- copy RoIs ground-truth data (will be manipulated, e.g. scaled)
    local img, rois = self:processImage(img, rois)
    img_size = img:size()        -- get final size
    if img_size[2] < 128 or img_size[3] < 128 then
      print(string.format("Warning: Skipping image '%s'. Invalid size after process: (%dx%d)", fn, img_size[3], img_size[2]))
      goto continue
    end

    table.insert(batch, { img = img, rois = rois })

    count = count - 1
    ::continue::
  end

  return batch
end
