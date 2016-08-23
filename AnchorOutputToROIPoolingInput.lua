require 'cunn'
require 'BatchIterator'
require 'Localizer'
require 'Anchors'
require 'Rect'

--local AnchorOutputToROIPoolingInput, Parent = torch.class('nn.AnchorOutputToROIPoolingInput', 'nn.Module')
local AnchorOutputToROIPoolingInput = torch.class('AnchorOutputToROIPoolingInput')

function AnchorOutputToROIPoolingInput:__init(model, testCase)
   --Parent.__init(self)
   self.model = model
   self.anchors = Anchors.new(model.pnet, model.cfg.scales)
   self.localizer = Localizer.new(model.pnet.outnode.children[#model.cfg.scales + 1])
   self.testCase = testCase or false
end


--function AnchorOutputToROIPoolingInput:updateOutput(input) -- input is pnet:forward(img)
function AnchorOutputToROIPoolingInput:detectRPI(input, img)

   -- inn.ROIPooling and also ROIPooling by fmassa is Spatial Adaptive Max Pooling (amp) Layer 
   -- for region proposals used in FastRCNN.
   -- Both expect a table on input, i.e. input = {data, rois}:
   --  * First argument data = input[1] is features in N x D x H x W, where 
   --     ** N = #images, 
   --     ** D = dim = #feature maps, 
   --     ** H = height, 
   --     ** W = width
   --    (i.e. 36 x 384 x 14 x 22)
   --  * Second argument rois = input[2] is bounding boxes in B x 5, where
   --     ** B = #bboxes,
   --     ** 5 = {imgId, minX, minY, maxX, maxY} in feature map coordinates

   local bboxTable = {} -- table of region proposals on feature map
   local softmax = nn.SoftMax():cuda()

   local matches = {} -- only for testing this class

   print('input:')
   print(input)
   print('img:size()')
   print(img:size())

   -- Compare "Detector.lua"
   -------------------------
   --print('self.model.cfg.scales:')
   --print(self.model.cfg.scales)
   --[[
   local scales = #self.model.cfg.scales -- 3
   local aspect_ratios = 3
   for i=1,scales do
     local layer = input[i]
     local layer_size = layer:size() -- z.B. 18 x 19 x 12
     for y=1,layer_size[2] do
       for x=1,layer_size[3] do
         for a=1,aspect_ratios do
   ]]
    -- only analyze anchors which are fully inside the input image
    local input_rect = Rect.new(0,0,img:size(3),img:size(2))
    local ranges = self.anchors:findRangesXY(input_rect, input_rect)
    for i,r in ipairs(ranges) do
      local layer = input[r.layer]
      local a = r.aspect
      for y=r.ly,r.uy-1 do
        for x=r.lx,r.ux-1 do

           local ofs = (a-1) * 6 -- offset
           local idx     = { { ofs + 1, ofs + 6 }, y, x } -- = { { a * 6 - 5, a * 6 }, y, x } -- 6 indices: cls (2) + reg (4)
           local cls_idx = { { ofs + 1, ofs + 2 }, y, x }
           local reg_idx = { { ofs + 3, ofs + 6 }, y, x }
           local scores = softmax:forward(layer[cls_idx])

           if scores[1] > 0.7 then -- scores[1]: score for positive, score[2]: score for negative

             -- layer[reg_idx] is the rect on the feature map of the i-th anchor net

             --[[local resultRectOnInput = Anchors.anchorToInput(self.anchors:get(i,a,y,x), layer[reg_idx]) -- region proposal]]
             local resultRectOnInput = Anchors.anchorToInput(self.anchors:get(r.layer,a,y,x), layer[reg_idx]) -- region proposal
             --print('resultRectOnInput:')
             --print(resultRectOnInput)

             local image_rect = Rect.new(0, 0, img:size(3), img:size(2))
             if resultRectOnInput:overlaps(image_rect) then
               table.insert(matches, { p=scores[1], r=resultRectOnInput }) -- only for testing this class

               -- Compare function "extract_roi_pooling_input" in "objective.lua":
               -------------------------------------------------------------------
               -- Transform RoI on input to RoI on feature map (FMRoI):
               local resultRectOnFeatureMap = self.localizer:inputToFeatureRect(resultRectOnInput, img, input[#input])
               --print('resultRectOnFeatureMap:')
               --print(resultRectOnFeatureMap)

               -- Clip FMRoI to feature map boundaries:
               local resultRectOnFeatureMap = resultRectOnFeatureMap:clip(Rect.new(0, 0, input[#input]:size(3), input[#input]:size(2)))
               --print('resultRectOnFeatureMap after clipping:')
               --print(resultRectOnFeatureMap)

               -- Start index has to be >=1 (0 is out of bound):
               local resultRect = Rect.new(math.min(math.max(resultRectOnFeatureMap.minX, 1), resultRectOnFeatureMap.maxX),
                                           math.min(math.max(resultRectOnFeatureMap.minY, 1), resultRectOnFeatureMap.maxY),
                                           resultRectOnFeatureMap.maxX,
                                           resultRectOnFeatureMap.maxY)
               --print('resultRect:')
               --print(resultRect)                              

               table.insert(bboxTable, {resultRect.minX, resultRect.minY, resultRect.maxX, resultRect.maxY})
               --print('bboxTable:')
               --print(bboxTable)
             end -- if resultRectOnInput:overlaps(image_rect) then
           end -- if scores[1] > 0.7 then
         --[[end -- for a=1,aspect_ratios do]]
       end -- for x=1,layer_size[3] do   bzw.  for x=r.lx,r.ux-1 do
     end -- for y=1,layer_size[2] do   bzw.  for y=r.ly,r.uy-1 do
   end -- for i=1,scales do          bzw.  for i,r in ipairs(ranges) do
   --print('#bboxTable:')
   --print(#bboxTable)
   --print('bboxTable:')
   --print(bboxTable)

   --self.output = bboxTable

   if self.testCase then
     --return self.output, matches
     return bboxTable, matches
   else
     --return self.output
     return bboxTable
   end
 
end


--function AnchorOutputToROIPoolingInput:updateGradInput(input, gradOutput)
--   -- (not implemented yet)
--   -- This layer does not propagate gradients.
--end

