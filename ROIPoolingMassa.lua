-- Copyright (c) 2015, Francisco Massa
-- All rights reserved.
--
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
--
-- * Redistributions of source code must retain the above copyright notice, this
--   list of conditions and the following disclaimer.
--
-- * Redistributions in binary form must reproduce the above copyright notice,
--   this list of conditions and the following disclaimer in the documentation
--   and/or other materials provided with the distribution.
--
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
-- AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
-- IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
-- DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
-- FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
-- DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
-- SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
-- CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
-- OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
-- OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-- Contact GitHub API Training Shop Blog About

local ROIPoolingMassa, parent = torch.class('nn.ROIPoolingMassa','nn.Module')

function ROIPoolingMassa:__init(W,H)
  parent.__init(self)
  self.W = W
  self.H = H
  self.pooler = {}--nn.SpatialAdaptiveMaxPooling(W,H)
  self.spatial_scale = 1
  self.gradInput = {torch.Tensor()}
end

function ROIPoolingMassa:setSpatialScale(scale)
  self.spatial_scale = scale
  return self
end

function ROIPoolingMassa:updateOutput(input)
  local data = input[1]
  local rois = input[2]

  local num_rois = rois:size(1)
  local s = data:size()
  local ss = s:size(1)
  self.output:resize(num_rois,s[ss-2],self.H,self.W)

  rois[{{},{2,5}}]:add(-1):mul(self.spatial_scale):add(1):round()
  rois[{{},2}]:cmin(s[ss])
  rois[{{},3}]:cmin(s[ss-1])
  rois[{{},4}]:cmin(s[ss])
  rois[{{},5}]:cmin(s[ss-1])

  -- element access is faster if not a cuda tensor
  if rois:type() == 'torch.CudaTensor' then
    self._rois = self._rois or torch.FloatTensor()
    self._rois:resize(rois:size()):copy(rois)
    rois = self._rois
  end

  if not self._type then self._type = self.output:type() end

  if #self.pooler < num_rois then
    local diff = num_rois - #self.pooler
    for i=1,diff do
      table.insert(self.pooler,nn.SpatialAdaptiveMaxPooling(self.W,self.H):type(self._type))
    end
  end

  for i=1,num_rois do
    local roi = rois[i]
    local im_idx = roi[1]
    local im = data[{im_idx,{},{roi[3],roi[5]},{roi[2],roi[4]}}]
    self.output[i] = self.pooler[i]:updateOutput(im)
  end
  return self.output
end

function ROIPoolingMassa:updateGradInput(input,gradOutput)
  local data = input[1]
  local rois = input[2]
  if rois:type() == 'torch.CudaTensor' then
    rois = self._rois
  end
  local num_rois = rois:size(1)
  local s = data:size()
  local ss = s:size(1)
  self.gradInput[1]:resizeAs(data):zero()

  for i=1,num_rois do
    local roi = rois[i]
    local im_idx = roi[1]
    local r = {im_idx,{},{roi[3],roi[5]},{roi[2],roi[4]}}
    local im = data[r]
    local g  = self.pooler[i]:updateGradInput(im,gradOutput[i])
    self.gradInput[1][r]:add(g)
  end
  return self.gradInput
end

function ROIPoolingMassa:type(type)
  parent.type(self,type)
  for i=1,#self.pooler do
    self.pooler[i]:type(type)
  end
  self._type = type
  return self
end
