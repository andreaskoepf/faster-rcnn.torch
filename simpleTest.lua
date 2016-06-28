--------------------------------------------------------------------------
-- Test, if our proposal net (without anchor nets) with weights copied
-- from the nn.Sequential part of the pretrained vgg16 net downloaded from: 
-- https://github.com/szagoruyko/fastrcnn-models.torch
-- generates the same features as the vgg16 net.
---------------------------------------------------------------------------
require 'torch'
require 'image'
require 'nngraph'
require 'cunn'


-- get configuration & model
----------------------------
cfg = dofile('config/imagenet.lua')
model_factory = dofile('models/pretrained/vgg16_ori.lua')
model = model_factory(cfg)

print(string.format('Number of nodes in model.pnet: %d', model.pnet:size()))
--graph.dot(model.pnet.fg, 'pnet', 'pnet')

local function get_net(net, from, to)
    local from = from or 1
    local to = to or 10
    local input_x = nn.Sequential()
    
    for j = from,to do
      for i = 1,#net:get(j) do
        input_x:add(net:get(j):get(i))
      end
    end
    
    return input_x
end

featureNet = get_net(model.pnet, 2, model.pnet:size()-#cfg.scales)


-- Load pretrained network:
----------------------------
local net = torch.load'/data/pretrained/vgg16_fast_rcnn_iter_40000.t7':unpack()
local seq = net:get(1):get(1)
print('nn.Sequential part of the pretrained network:')
print(seq)
print('Corresponding part of our featureNet:')
print(featureNet)


-- Test with lena image:
-------------------------
lena = image.lena()
lena = image.scale(lena, 224, 224)
output_featureNet = featureNet:forward(lena)
output_seq = seq:forward(lena:cuda())

print('output_featureNet-output_seq:')
print(torch.norm(output_featureNet:cuda()-output_seq))

-- Test with random input tensor:
----------------------------------
img = torch.rand(3,200,300)
outA = featureNet:forward(img)
outB = seq:forward(img:cuda())
B = outA:cuda() - outB
print(B:norm())
assert(B:norm() < 1e-03, "output is not equal!!")
print("DONE!!")
