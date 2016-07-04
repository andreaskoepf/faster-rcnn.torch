 local coco_cfg = {
  class_count = 80,  -- excluding background class
  target_smaller_side = 224,
  scales = { 48, 96, 192, 384 },
  max_pixel_size = 1000,
  normalization = { },
  augmentation = { vflip = 0, hflip = 0.25, random_scaling = 0, aspect_jitter = 0 },
  color_space = 'rgb',
  roi_pooling = { kw = 6, kh = 6 },
  examples_base_path = '/data/coco/images/val2014/',
  background_base_path = '',
  batch_size = 256,
  positive_threshold = 0.7,
  negative_threshold = 0.3,
  best_match = true,
  nearby_aversion = true
}

return coco_cfg
