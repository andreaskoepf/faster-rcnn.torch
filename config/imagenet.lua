 local imgnet_cfg = {
  class_count = 200,  -- excluding background class
  target_smaller_side = 480,
  scales = { 48, 96, 192, 384 },
  max_pixel_size = 1000,
  normalization = { method = 'none' },
  augmentation = { vflip = 0, hflip = 0.25, random_scaling = 0, aspect_jitter = 0 },
  color_space = 'yuv',
  roi_pooling = { kw = 6, kh = 6 },
  examples_base_path = '',
  background_base_path = '',
  batch_size = 300,
  positive_threshold = 0.6, 
  negative_threshold = 0.25,
  best_match = true,
  nearby_aversion = true
}

return imgnet_cfg