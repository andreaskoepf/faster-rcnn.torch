 local duplo_cfg = {
  class_count = 16,  -- excluding background class
  target_smaller_side = 450,
  --scales = { 48, 96, 192, 384 },
  scales = { 32, 64, 128, 256 },
  max_pixel_size = 1000,
  --normalization = { method = 'contrastive', width = 7 },
  normalization = { method = 'none' },
  augmentation = { vflip = 0.5, hflip = 0.5, random_scaling = 0.5, aspect_jitter = 0.2 },
  color_space = 'yuv',
  roi_pooling = { kw = 6, kh = 6 },
  examples_base_path = '/home/koepf/datasets/brickset_all/',
  background_base_path = '/home/koepf/datasets/background/',
  batch_size = 256,
  positive_threshold = 0.5, 
  negative_threshold = 0.25,
  best_match = true,
  nearby_aversion = true
}

return duplo_cfg