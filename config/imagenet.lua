 local imgnet_cfg = {
  class_count = 200,  -- excluding background class
  target_smaller_side = 300,
  scales = { 32, 128, 256 },
  max_pixel_size = 500,
  --normalization = { method = 'contrastive', width = 7, centering = true, scaling = true },
  normalization = { },
  augmentation = { vflip = 0, hflip = 0.2, random_scaling = 0, aspect_jitter = 0 },
  color_space = 'rgb',
  roi_pooling = { kw = 3, kh = 3},
  examples_base_path = '',
  background_base_path = '',
  batch_size = 256,
  positive_threshold = 0.7, 
  negative_threshold = 0.3,
  best_match = false,
  nearby_aversion = false
}

return imgnet_cfg