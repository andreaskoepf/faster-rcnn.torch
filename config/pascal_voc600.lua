 local pascal_voc_cfg = {
  class_count = 20,  -- excluding background class
  target_smaller_side = 600,
  scales = { 96, 172, 344 },
  max_pixel_size = 1000,
  normalization = {},--{ centering = false, scaling = false },
  augmentation = { vflip = 0.0, hflip = 0.0, random_scaling = 0.0, aspect_jitter = 0.0 },
  color_space = 'rgb',
  roi_pooling = { kw = 3, kh = 3 },
  examples_base_path = '/data/VOC2007/',
  background_base_path = nil,
  batch_size = 256,
  positive_threshold = 0.7, 
  negative_threshold = 0.3,
  best_match = false,
  nearby_aversion = false,
  uniform_image_scaling = true, 
  backgroundClass = 1
}

return pascal_voc_cfg
