require 'models.model_utilities'

-- layer here means a block of one or more convolution layers optionally followed by pooling layer
local layers = {
  { filters= 64, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=2, pooling='max' },
  { filters=128, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=2, pooling='max' },
  { filters=256, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=3, pooling='max' },
  { filters=512, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=3, pooling='max' },
  { filters=512, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=3, pooling='none' }
}

local anchor_nets = {
  { kW=3, n=256, input=5 }, -- input refers to the 'layer' defined above
  { kW=3, n=256, input=5 },
  { kW=3, n=256, input=5 }
}

local class_layers =  {
  { n=1024, dropout=0.5, batch_norm=true },
  { n= 512, dropout=0.5 },
}

-- network factory function
function vgg_small(cfg)
  return create_model(cfg, layers, anchor_nets, class_layers)
end

return vgg_small, layers, anchor_nets, class_layers
