require 'cunn'
require 'BatchIterator'
require 'Localizer'
require 'Anchors'
require 'Rect'

local AnchorOutputToROIPoolingInput, Parent = torch.class('nn.AnchorOutputToROIPoolingInput', 'nn.Module')


function AnchorOutputToROIPoolingInput:__init(model)
   Parent.__init(self)
   self.model = model
   self.anchors = Anchors.new(model.pnet, model.cfg.scales)
   self.localizer = Localizer.new(model.pnet.outnode.children[5])
end


function AnchorOutputToROIPoolingInput:updateOutput(input) -- input is pnet:forward(img)

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

   --print('Feature map size:')
   --print(input[5]:size())

   -- Compare "Detector.lua"
   -------------------------
   local scales = 4
   local aspect_ratios = 3
   for i=1,scales do
     local layer = input[i]
     local layer_size = layer:size() -- z.B. 18 x 19 x 12
     for y=1,layer_size[2] do
       for x=1,layer_size[3] do
         for a=1,aspect_ratios do

           local ofs = (a-1) * 6 -- offset
           local idx     = { { ofs + 1, ofs + 6 }, y, x } -- = { { a * 6 - 5, a * 6 }, y, x } -- 6 indices: cls (2) + reg (4)
           local cls_idx = { { ofs + 1, ofs + 2 }, y, x }
           local reg_idx = { { ofs + 3, ofs + 6 }, y, x }
           local scores = softmax:forward(layer[cls_idx])
           local resultRectOnInput = Anchors.anchorToInput(self.anchors:get(i,a,y,x), layer[reg_idx]) -- region proposal
           --print('resultRectOnInput:')
           --print(resultRectOnInput)

           -- Compare function "extract_roi_pooling_input" in "objective.lua":
           -------------------------------------------------------------------
           -- Transform RoI on input to RoI on feature map (FMRoI):
           local resultRectOnFeatureMap = self.localizer:inputToFeatureRect(resultRectOnInput)
           --print('resultRectOnFeatureMap:')
           --print(resultRectOnFeatureMap)

           -- Clip FMRoI to feature map boundaries:
           resultRectOnFeatureMap = resultRectOnFeatureMap:clip(Rect.new(0, 0, input[5]:size()[3], input[5]:size()[2]))
           --print('resultRectOnFeatureMap after clipping:')
           --print(resultRectOnFeatureMap)

           -- Start index has to be >=1 (0 is out of bound):
           local resultRect = Rect.new(math.min(resultRectOnFeatureMap.minX + 1, resultRectOnFeatureMap.maxX),
                                       math.min(resultRectOnFeatureMap.minY + 1, resultRectOnFeatureMap.maxY),
                                       resultRectOnFeatureMap.maxX,
                                       resultRectOnFeatureMap.maxY)
           --print('resultRect:')
           --print(resultRect)

           if scores[1] > 0.7 then  -- scores[1]: score for positive, score[2]: score for negative
             table.insert(bboxTable, {resultRect.minX, resultRect.minY, resultRect.maxX, resultRect.maxY})
             --print('bboxTable:')
             --print(bboxTable)
           end
         end
       end  
     end
   end
   --print('#bboxTable:')
   --print(#bboxTable)
   --print('bboxTable:')
   --print(bboxTable)

   self.output = bboxTable
   return self.output

end


function AnchorOutputToROIPoolingInput:updateGradInput(input, gradOutput)
   -- not implemented yet
end

