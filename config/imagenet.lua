 local imgnet_cfg = {
  class_count = 200,  -- excluding background class
  target_smaller_side = 224,
  scales = { 48, 96, 172 },
  max_pixel_size = 1000,
  normalization = { },
  augmentation = { }, --{ vflip = 0, hflip = 0.2, random_scaling = 0, aspect_jitter = 0 },
  color_space = 'rgb',
  roi_pooling = { kw = 3, kh = 3},
  examples_base_path = '',
  background_base_path = '',
  batch_size = 256,
  positive_threshold = 0.7, 
  negative_threshold = 0.3,
  best_match = false,
  nearby_aversion = true
}

return imgnet_cfg
