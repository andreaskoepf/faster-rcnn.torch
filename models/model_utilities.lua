require 'nngraph'

function create_proposal_net(layers, anchor_nets)
  -- define  building block functions first

  -- VGG style 3x3 convolution building block
  local function ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout)
    container:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW,kH, 1,1, padW,padH))
    container:add(nn.PReLU())
    if dropout then
      container:add(nn.Dropout(dropout))
    end
    return container
  end
  
  -- multiple convolution layers followed by a max-pooling layer
  local function ConvPoolBlock(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, conv_steps)
    for i=1,conv_steps do
      ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout)
      nInputPlane = nOutputPlane
    end
    container:add(nn.SpatialMaxPooling(2, 2, 2, 2):ceil())
    return container
  end  
  
  -- creates anchor network which reduces to a 256 dimensional vector and 
  -- then to anchors outputs for 3 aspect ratios 
  local function AnchorNetwork(nInputPlane, kernelWidth)
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(nInputPlane, 256, kernelWidth,kernelWidth, 1,1))
    net:add(nn.PReLU())
    net:add(nn.SpatialConvolution(256, 3 * (2 + 4), 1, 1))  -- aspect ratios { 1:1, 2:1, 1:2 } x { class, left, top, width, height }
    return net
  end

  local input = nn.Identity()()
    
  local conv_outputs = {}
  
  local nInputPlane = 3
  local prev = input
  for i,l in ipairs(layers) do
    local net = nn.Sequential()
    ConvPoolBlock(net, nInputPlane, l.filters, l.kW, l.kH, l.padW, l.padH, l.dropout, l.conv_steps)
    nInputPlane = l.filters
    prev = net(prev)
    table.insert(conv_outputs, prev)
  end
  
  local anchor_nets = {
    { kW=3, n=256, input=3 },   -- input refers to the 'layer' defined above
    { kW=3, n=256, input=4 },
    { kW=5, n=256, input=4 },
    { kW=7, n=256, input=4 }
  }
  
  -- split the main network into two parts because we build 
  -- region proposals for the smallest scale after 3 max-pooling steps (with a input stride of 8 instead of 16)
    
  local convout1 = net1(input)
  local convout2 = net2(convout1)
  local a1 = AnchorNetwork(200, 3)(convout1)
  local a2 = AnchorNetwork(300, 3)(convout2)
  local a3 = AnchorNetwork(300, 5)(convout2)
  local a4 = AnchorNetwork(300, 7)(convout2)
  
    -- create multi-output module
  local model = nn.gModule({ input }, { a1, a2, a3, a4, convout2 })
  
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

function create_classifaction_net(kw, kh, planes, class_count)
  -- create classifiaction network
  local net = nn.Sequential()
  
  net:add(nn.Linear(kh * kw * planes, 1024))
  net:add(nn.PReLU())
  net:add(nn.Dropout(0.5))
  net:add(nn.Linear(1024, 1024))
  net:add(nn.PReLU())
  net:add(nn.Dropout(0.5))
  
  local input = nn.Identity()()
  local node = net(input)
  
  -- now the network splits into regression and classification branches
  
  -- regression output
  local rout = nn.Linear(1024, 4)(node)
  
  -- classification output
  local cnet = nn.Sequential()
  cnet:add(nn.Linear(1024, class_count))
  cnet:add(nn.LogSoftMax())
  local cout = cnet(node)
  
  -- create multi-output module
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

function create_model(cfg, layers, anchor_nets)
  local model = 
  {
    cfg = cfg,
    pnet = create_proposal_net(layers, anchor_nets),
    cnet = create_classification_net(cfg.roi_pooling.kw, cfg.roi_pooling.kh, 300, cfg.class_count + 1)
  }
  return model
end
