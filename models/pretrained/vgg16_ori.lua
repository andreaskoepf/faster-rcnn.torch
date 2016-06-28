require 'models.model_utilities'
local cudnn = require 'cudnn'
local inn = require 'inn'

function vgg16_ori(cfg)
  -- layer here means a block of one or more convolution layers followed by a max-pooling layer
  local layers = {
    { filters= 64, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=2 },
    { filters=128, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=2 },
    { filters=256, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=3 },
    { filters=512, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=3 },
    { filters=512, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=3 } -- In the last block the max-pooling layer will be removed.
  }

  local anchor_nets = {
    { kW=3, n=512, input=5 },   -- input refers to the 'layer' defined above
    { kW=3, n=256, input=5 },
    { kW=5, n=256, input=5 },
    { kW=7, n=256, input=5 }
  }

  local class_layers =  {
    { n=1024, dropout=0.5, batch_norm=true },
    { n=512, dropout=0.5 }
  }
  model = create_model(cfg, layers, anchor_nets, class_layers)
  model.pnet:get(#layers+1):remove(7)

  print(model.pnet:get(#layers+1))
  
  -- Load pretrained network:
  ----------------------------
  local net = torch.load'/data/pretrained/vgg16_fast_rcnn_iter_40000.t7':unpack()
  --print('pretrained network:')
  --print(net)

  local counter = 1
  local seq = net:get(1):get(1)
  print('nn.Sequential part of the net:')
  print(seq)

  -- Take the spatial convolution weights from the pretrained network
  local name_nn ='nn.SpatialConvolution'
  local name_cudnn ='cudnn.SpatialConvolution'
  
  local v1 = seq:findModules(name_cudnn)
  print('cudnn.SpatialConvolution nodes:')
  print(v1)
  print('number of cudnn.SpatialConvolution nodes (pretrained network):')
  print(#v1)

  for k,v in pairs(model.pnet:findModules(name_nn)) do
    print('nn.SpatialConvolution node of this net:')
    print(v)
    print('Number of weights:')
    print(v.weight:size())
    print('Respective cudnn.SpatialConvolution node of pretrained vgg16:')
    print(v1[counter])
    print('Number of weights:')
    print(v1[counter].weight:size())
    v.weight:copy(v1[counter].weight)
    v.bias:copy(v1[counter].bias)
    counter = counter + 1
    if counter > #v1 then
      break
    end
  end

  return model
end

return vgg16_ori
