 local imgnet_cfg = {
  class_count = 200,  -- excluding background class
  target_smaller_side = 224,
  uniform_image_scaling = true,
  scales = { 48, 96, 172 },
  max_pixel_size = 1000,
  -- normalization = { scaling = true, centering = true, method = 'contrastive', width = 7 },
  normalization = { },
  augmentation = { vflip = 0, hflip = 0.25, random_scaling = 0, aspect_jitter = 0 },
  color_space = 'rgb',
  roi_pooling = { kw = 6, kh = 6 },
  examples_base_path = '',
  background_base_path = '',
  batch_size = 256,
  positive_threshold = 0.7,
  negative_threshold = 0.3,
  best_match = true,
  nearby_aversion = false,
  backgroundClass = 201
}

return imgnet_cfg
