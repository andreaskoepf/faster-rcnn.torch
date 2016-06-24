require 'models.model_utilities'

-- layer here means a block of one or more convolution layers followed by a max-pooling layer
local layers = {
  { filters= 64, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=1 },
  { filters=128, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=2 },
  { filters=256, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=2 },
  { filters=384, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=2 }
}

local anchor_nets = {
--  { kW=3, n=256, input=3 },   -- input refers to the 'layer' defined above
  { kW=4, n=256, input=4 },
  { kW=4, n=256, input=4 },
  { kW=4, n=256, input=4 }
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
