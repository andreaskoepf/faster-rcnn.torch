local ROIPooling, parent = torch.class('nn.ROIPooling','nn.Module')

function ROIPooling:__init(W,H)
  parent.__init(self)
  self.W = W
  self.H = H
  self.pooler = {}--nn.SpatialAdaptiveMaxPooling(W,H)
  self.spatial_scale = 1
  self.gradInput = {torch.Tensor()} -- TODO: gradInput has variable length inside. 
end

function ROIPooling:setSpatialScale(scale)
  self.spatial_scale = scale
  return self
end

function ROIPooling:updateOutput(input)
  local data = input.data
  local rois = input.rois

  local num_rois = #rois
  local mySize = data[1].featureMap:size(1)
  self.output:resize(num_rois,mySize,self.H,self.W) -- #bboxes x 384 x 6 x 6
  --print('self.output:size()')
  --print(self.output:size())

  --print('rois[1]:')
  --print(rois[1])
  --print('rois[1].minX:')
  --print(rois[1].minX)
  --print('rois[1].minY:')
  --print(rois[1].minY)
  --print('rois[1].maxX:')
  --print(rois[1].maxX)
  --print('rois[1].maxY:')
  --print(rois[1].maxY)

  for i=1,#rois do
    rois[i].minX = math.floor(((rois[i].minX - 1) * self.spatial_scale + 1) + 0.5)
    rois[i].minY = math.floor(((rois[i].minY - 1) * self.spatial_scale + 1) + 0.5)
    rois[i].maxX = math.floor(((rois[i].maxX - 1) * self.spatial_scale + 1) + 0.5)
    rois[i].maxY = math.floor(((rois[i].maxY - 1) * self.spatial_scale + 1) + 0.5)
    rois[i].minX = math.min(rois[i].minX, data[rois[i].imgID].featureMap:size(3))
    rois[i].minY = math.min(rois[i].minY, data[rois[i].imgID].featureMap:size(2))
    rois[i].maxX = math.min(rois[i].maxX, data[rois[i].imgID].featureMap:size(3))
    rois[i].maxY = math.min(rois[i].maxY, data[rois[i].imgID].featureMap:size(2))
  end

  if not self._type then self._type = self.output:type() end

  if #self.pooler < num_rois then
    local diff = num_rois - #self.pooler
    for i=1,diff do
      table.insert(self.pooler,nn.SpatialAdaptiveMaxPooling(self.W,self.H):type(self._type))
    end
  end

  --print('data[1]:')
  --print(data[1])

  for i=1,num_rois do
    local roi = rois[i]
    --print(string.format('iteration: %d', i))
    --print('roi:')
    --print(roi)
    local im_idx = roi.imgID
    --print(string.format('data[%d].featureMap:size():', im_idx))
    --print(data[im_idx].featureMap:size())
    local im = data[im_idx].featureMap[{{},{roi.minY,roi.maxY},{roi.minX,roi.maxX}}]
    --print('im:size():')
    --print(im:size())
    self.output[i] = self.pooler[i]:updateOutput(im)
    --print('self.output[i]:size():')
    --print(self.output[i]:size())
  end
  return self.output
end

-- TODO: This has to be implemented correctly, such that it works with
--       different feature map sizes for different images.
function ROIPooling:updateGradInput(input,gradOutput)
  local data = input.data
  local rois = input.rois
  --if rois:type() == 'torch.CudaTensor' then
  --  rois = self._rois
  --end
  local num_rois = #rois
  local mySize = data[1].featureMap:size(1)
  -- TODO: The feature maps for different images have different sizes.
  --       -> gradInput has to be of variable size (i.e. a table?)
  --self.gradInput[1]:resizeAs(data):zero() -- This won't work, because data is a table of
                                            -- image IDs and feature maps of different sizes.
  local num_imgs = #data
  --self.gradInput[1]:resize(num_imgs,mySize,???) -- #imgs x 384 x featureMap-sizes

  for i=1,num_rois do
    local roi = rois[i]
    local im_idx = roi.imgID
    --local r = {im_idx,{},{roi[3],roi[5]},{roi[2],roi[4]}}
    local im = data[im_idx].featureMap[{{},{roi.minY,roi.maxY},{roi.minX,roi.maxX}}]
    local g  = self.pooler[i]:updateGradInput(im,gradOutput[i])
    --self.gradInput[1][r]:add(g) -- This won't work.
    table.insert(gradInput[1], {img_idx, g}) --???
  end
  return self.gradInput
end

function ROIPooling:type(type)
  parent.type(self,type)
  for i=1,#self.pooler do
    self.pooler[i]:type(type)
  end
  self._type = type
  return self
end
