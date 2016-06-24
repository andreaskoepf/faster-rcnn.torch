require 'nngraph'

function create_conv_layers(layers, input)
  -- VGG style 3x3 convolution building block
  local function ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, bn)
    container:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW,kH, 1,1, padW,padH))
    if bn then
      container:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
    end
    container:add(nn.PReLU())
    if dropout and dropout > 0 then
      container:add(nn.SpatialDropout(dropout))
    end
    return container
  end

  -- multiple convolution layers followed by a max-pooling layer
  local function ConvPoolBlock(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, conv_steps)
    local bn = false
    for i=1,conv_steps do
      ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, bn)
      nInputPlane = nOutputPlane
      dropout = nil -- only one dropout layer per conv-pool block
      bn = false
    end
    container:add(nn.SpatialMaxPooling(2, 2, 2, 2):ceil())
    return container
  end

  local conv_outputs = {}

  local inputs = 3
  local prev = input
  for i,l in ipairs(layers) do
    local net = nn.Sequential()
    ConvPoolBlock(net, inputs, l.filters, l.kW, l.kH, l.padW, l.padH, l.dropout, l.conv_steps)
    inputs = l.filters
    prev = net(prev)
    table.insert(conv_outputs, prev)
  end

  return conv_outputs
end

function gaussian_init(module, name)
  local function init_module(m)
    for k,v in pairs(m:findModules(name)) do
      local n = v.kW * v.kH * v.nOutputPlane
      v.weight:normal(0, math.sqrt(2 / n))
      v.bias:zero()
    end
  end
  module:apply(init_module)
end

function create_simple_pretraining_net(layers, conv_output_count, class_count, fc_size)
  local input = nn.Identity()()
  local conv_outputs = create_conv_layers(layers, input)

  -- add alexnet like FC layers
  fc_size = fc_size or 512
  local classifier = nn.Sequential()

  classifier:add(nn.View(conv_output_count))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(conv_output_count, fc_size))
  classifier:add(nn.ReLU(true))

  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(fc_size, fc_size))
  classifier:add(nn.ReLU(true))

  classifier:add(nn.Linear(fc_size, class_count))
  classifier:add(nn.LogSoftMax())

  local classifier_output = classifier(conv_outputs[#conv_outputs])

  local model = nn.gModule({ input }, { classifier_output })
  gaussian_init(model, 'nn.SpatialConvolution')
  return model
end

function create_proposal_net(layers, anchor_nets)

  -- creates an anchor network which reduces the input first to 256 dimensions
  -- and then further to the anchor outputs for 3 aspect ratios
  local function AnchorNetwork(nInputPlane, n, kernelWidth)
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(nInputPlane, n, kernelWidth,kernelWidth, 1,1))
    net:add(nn.PReLU())
    net:add(nn.SpatialConvolution(n, 3 * (2 + 4), 1, 1))  -- aspect ratios { 1:1, 2:1, 1:2 } x { class, left, top, width, height }
    return net
  end

  local input = nn.Identity()()
  local conv_outputs = create_conv_layers(layers, input)

  local proposal_outputs = {}
  for i,a in ipairs(anchor_nets) do
    table.insert(proposal_outputs, AnchorNetwork(layers[a.input].filters, a.n, a.kW)(conv_outputs[a.input]))
  end
  table.insert(proposal_outputs, conv_outputs[#conv_outputs])

    -- create proposal net module, outputs: anchor net outputs followed by last conv-layer output
  local model = nn.gModule({ input }, proposal_outputs)
  gaussian_init(model, 'nn.SpatialConvolution')
  return model
end

function create_classification_net(inputs, class_count, class_layers)
  -- create classifiaction network
  local net = nn.Sequential()

  local prev_input_count = inputs
  for i,l in ipairs(class_layers) do
    net:add(nn.Linear(prev_input_count, l.n))
    if l.batch_norm then
      net:add(nn.BatchNormalization(l.n))
    end
    net:add(nn.PReLU())
    if l.dropout and l.dropout > 0 then
      net:add(nn.Dropout(l.dropout))
    end
    prev_input_count = l.n
  end

  local input = nn.Identity()()
  local node = net(input)

  -- now the network splits into regression and classification branches

  -- regression output
  local rout = nn.Linear(prev_input_count, 4)(node)

  -- classification output
  local cnet = nn.Sequential()
  cnet:add(nn.Linear(prev_input_count, class_count))
  cnet:add(nn.LogSoftMax())
  local cout = cnet(node)

  -- create bbox finetuning + classification output
  local model = nn.gModule({ input }, { rout, cout })

  local function init(module, name)
    local function init_module(m)
      for k,v in pairs(m:findModules(name)) do
        local n = v.kW * v.kH * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2 / n))
        v.bias:zero()
      end
    end
    module:apply(init_module)
  end

  init(model, 'nn.SpatialConvolution')

  return model
end

function create_model(cfg, layers, anchor_nets, class_layers)
  local cnet_ninputs = cfg.roi_pooling.kh * cfg.roi_pooling.kw * layers[#layers].filters
  local model =
  {
    cfg = cfg,
    layers = layers,
    pnet = create_proposal_net(layers, anchor_nets),
    cnet = create_classification_net(cnet_ninputs, cfg.class_count + 1, class_layers)
  }
  return model
end
