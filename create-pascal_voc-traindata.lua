require 'torch'
require 'LuaXML'
require 'utilities'
require 'Rect'
require 'lfs'
local matio = require 'matio'

local function merge_roidbs(a, b)
  assert(#a == #b)
  for i = 1,#a do
    torch.cat(a[i].boxes,b[i].boxes,1)
    torch.cat(a[i].gt_classes, b[i].gt_classes,1)
    torch.cat(a[i].gt_overlaps, b[i].gt_overlaps,1)
  end
  return a
end

local pascal_voc = torch.class('Pascal_voc')
function pascal_voc:__init(image_set, year, devkit_path)
  self.year = year
  self.image_set = image_set
  self.devkit_path = devkit_path or '/data/VOCdevkit'
  self.data_path = string.format('%s/VOC%d',self.devkit_path, self.year)
  self.classes = {'__background__', -- always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'}
  self.num_classes = #self.classes
  self.class_to_ind = {}
  for i,v in pairs(self.classes) do
    self.class_to_ind[v] = i 
  end
  self.image_ext = '.jpg'
  self.image_index = self:load_image_set_index()
  --self.salt = str(uuid.uuid4())
  self.comp_id = 'comp4'
  self.name = string.format("voc_%d_%s",self.year,self.image_set)
  self.cache_path = string.format("%s/cache",self.data_path)
  if not path.exists(self.cache_path) then
    lfs.mkdir(self.cache_path)
  end
  -- PASCAL specific config options
  self.config = {cleanup = true, use_salt = true, top_k = 2000,use_diff = false, rpn_file = nil}

  assert(path.exists(self.devkit_path), string.format('VOCdevkit path does not exist: %s',self.devkit_path))
  assert(path.exists(self.data_path), string.format('Path does not exist: %s',self.data_path))
  
  if true then
    self.roidb_handler = self.selective_search_roidb
    self.roidb = self:selective_search_roidb()
  else
    self.roidb_handler = self.gt_roidb
    self.roidb = self:gt_roidb()
  end
  
end

function pascal_voc:num_images()
  return #self.image_index
end

function pascal_voc:bbox_overlaps(box1, box2)
  local num_boxes = box2:size(1)
  local overlabs = torch.Tensor(num_boxes)
  if box1:size(1) == 1 and box2:size(1) > 1 then
    for i = 1, num_boxes do
      local b1 = Rect(box1[{1,1}],box1[{1,2}],box1[{1,3}],box1[{1,4}])
      local b2 = Rect(box2[{i,1}],box2[{i,2}],box2[{i,3}],box2[{i,4}])
      overlabs[i] = b1:IoU(b2)
    end
  else
    for i = 1, num_boxes do
      local b1 = Rect(box1[{i,1}],box1[{i,2}],box1[{i,3}],box1[{i,4}])
      local b2 = Rect(box2[{i,1}],box2[{i,2}],box2[{i,3}],box2[{i,4}])
      overlabs[i] = b1:IoU(b2)
    end
  end
  return overlabs
end

function pascal_voc:create_roidb_from_box_list(box_list, gt_roidb)
  assert(#box_list == self:num_images(), 'Number of boxes must match number of ground-truth images')
  local roidb = {}
  for i = 1,self:num_images() do
      xlua.progress(i, self:num_images())
      local boxes = box_list[i]
      
      local num_boxes = boxes:size(1)
      
      local overlaps = torch.zeros(num_boxes, self.num_classes)

      if gt_roidb then
          local gt_boxes = gt_roidb[i].boxes
          --print(gt_boxes)
          local gt_classes = gt_roidb[i].gt_classes
          for j = 1,num_boxes do
            local gt_overlaps = self:bbox_overlaps(boxes[{{j},{}}], gt_boxes)
            local maxes, argmaxes = torch.max(gt_overlaps,1)
            local I = maxes:gt(0) 
            
            if I[1]>0 then
             overlaps[{{I[1]}, {gt_classes[argmaxes[I[1]]]}}] = maxes[I[1]]
            end
          end
      end
      --overlaps = scipy.sparse.csr_matrix(overlaps)
      table.insert(roidb,{boxes = boxes,
                    gt_classes = torch.zeros(num_boxes),
                    gt_overlaps= overlaps,
                    flipped = false})
  end
  return roidb
end
function pascal_voc:image_path_from_index(index)
  local image_path = string.format("%s/JPEGImages/%06d.jpg",self.data_path, index)
  assert(path.exists(image_path), string.format('Path does not exist: %s',image_path))
  return image_path
end

function pascal_voc:image_path_at(image_index)
  return self:image_path_from_index(image_index)
end

function pascal_voc:load_image_set_index()
  local root_path = self.data_path
  -- Example path to image set file:
  -- devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
  local image_set_file = string.format("%s/ImageSets/Main/%s.txt", root_path,self.image_set)
  assert(path.exists(image_set_file), string.format('Path does not exist: %s', image_set_file))
  local image_index={}
  local f = io.open(image_set_file, 'r')
  for l in f:lines() do
    local v = l:split(' ')
    image_index[#image_index + 1] = tonumber(v[1])
  end
  return image_index
end

function pascal_voc:gt_roidb()
  print("Return the database of ground-truth regions of interest. This function loads/saves from/to a cache file to speed up future calls.")
  local cache_file = string.format("%s/%s_gt_roidb.t7", self.cache_path, self.name)
  if path.exists(cache_file) then
    local roidb=torch.load(cache_file)
    print( string.format('%s gt roidb loaded from %s.' ,self.name, cache_file))
    return roidb
  end

  local gt_roidb = {}
  
  for index,value in ipairs(self.image_index) do
    gt_roidb[#gt_roidb + 1] = self:load_pascal_annotation(value)
  end
  torch.save(cache_file,gt_roidb)
  print(string.format('wrote gt roidb to %s.',cache_file))

  return gt_roidb
end
    
function pascal_voc:load_selective_search_roidb( gt_roidb)
        local filename = string.format('%s/selective_search_data/%s.mat',self.data_path,self.name) --original it is mat

        assert( path.exists(filename), string.format('Selective search data not found at: %s',filename))
        --local raw_data = sio.loadmat(filename)['boxes'].ravel()
        local raw_data = matio.load(filename)
        
        --print(string.format('[load_selective_search_roidb] size raw_data boxes = %d',#raw_data.boxes))
        
        local box_list = {}
        for i =1,#raw_data.boxes do
         local B = raw_data.boxes[i] --representation is in x y w h need to be minx maxx miny maxy
         local minx =B[{{},{1}}]
         local maxx =B[{{},{1}}]+B[{{},{3}}]
         local miny =B[{{},{2}}]
         local maxy =B[{{},{2}}]+B[{{},{4}}]
         local C1 =torch.cat(minx,miny)
         local C2 =torch.cat(maxx,maxy)
         B = torch.cat(C1,C2)
          table.insert(box_list, B)
        end
        print(string.format('[load_selective_search_roidb] size box_list = %d',#box_list))
  return self:create_roidb_from_box_list(box_list, gt_roidb)
end


function pascal_voc:selective_search_roidb()
  print("Return the database of selective search regions of interest.\nGround-truth ROIs are also included.\nThis function loads/saves from/to a cache file to speed up future calls.")
  local cache_file = string.format("%s/%s_selective_search_roidb.t7", self.cache_path, self.name)

  if path.exists(cache_file) then
    local roidb=torch.load(cache_file)
    print(string.format('%s ss roidb loaded from %s',self.name, cache_file))
    return roidb
  end
  
  local gt_roidb, ss_roidb, roidb
  if (self.year == 2007) or (self.image_set ~= 'test') then
    gt_roidb = self:gt_roidb()
    ss_roidb = self:load_selective_search_roidb(gt_roidb)
    roidb = merge_roidbs(gt_roidb, ss_roidb)
  else
    roidb = self:load_selective_search_roidb()
  end
  torch.save(cache_file,roidb)
  print(string.format('wrote ss roidb to %s}',cache_file))
  return roidb
end


function pascal_voc:rpn_roidb()
  local gt_roidb, ss_roidb, roidb,rpn_roidb
  if self.year == 2007 or self.image_set ~= 'test' then
    gt_roidb = self:gt_roidb()
    rpn_roidb = self:load_rpn_roidb(gt_roidb)
    roidb.insert(rpn_roidb) --= imdb.merge_roidbs(gt_roidb, rpn_roidb)
  else
    roidb = self:load_rpn_roidb()
  end

  return roidb
end

function pascal_voc:load_rpn_roidb(gt_roidb)
  local filename = self.config.rpn_file or "pascal_voc_2007.t7"
  assert(path.exists(filename), string.format('rpn data not found at: %s', filename))
  print(string.format('loading %s',filename))
  local box_list = torch.load(filename)
  return self:create_roidb_from_box_list(box_list, gt_roidb)
end

function pascal_voc:load_pascal_annotation(index)
  print("Load image and bounding boxes info from XML file in the PASCAL VOC")
  local num_objs = 1
  local filename = string.format('%s/Annotations/%06d.xml', self.data_path, index)

  local xfile = xml.load(filename)
  local finalObjects = {}
  local boxes
  local gt_classes
  local overlaps
  local bbox,x1,x2,y1,y2
  local non_diff_objs = {}
  --print(xfile)
  for _,objs in pairs(xfile) do
    if xml.find(objs,'object') then
      if not self.config.use_diff then
        --Exclude the samples labeled as difficult
        for _,obj in pairs(objs) do
          if xml.find(obj, "difficult") then
            if tonumber(obj[1]) == 0 then
              non_diff_objs[#non_diff_objs + 1]= objs
            end
          end
        end
        if #non_diff_objs ~= #xfile then
          print(string.format('Removed %d difficult objects', #xfile - #non_diff_objs))
        end
      else
        --Exclude the samples labeled as difficult
        for _,obj in ipairs(objs) do
          finalObjects[#finalObjects + 1]= objs
        end
      end
    end
  end
   if not self.config.use_diff then
    finalObjects = non_diff_objs
   end
  num_objs = #finalObjects
  boxes = torch.zeros(num_objs, 4)
  gt_classes = torch.zeros(num_objs)
  overlaps = torch.zeros(num_objs, self.num_classes)
  -- Load object bounding boxes into a data frame.
  for ix,objs in ipairs(finalObjects) do
    local name = xml.find(objs,'name')
     local cls = self.class_to_ind[name[1]]
     if not cls then
        break
     end
    if xml.find(objs,'object') then
      for _, obj in ipairs(objs) do
          bbox = obj:find('bndbox')
          if bbox then 
            x1 = tonumber(bbox:find('xmin')[1])
            x2 = tonumber(bbox:find('xmax')[1])
            y1 = tonumber(bbox:find('ymin')[1])
            y2 = tonumber(bbox:find('ymax')[1])
            boxes[{ix, {}}] = torch.Tensor({x1, y1, x2, y2})
            gt_classes[ix] = cls
            overlaps[{ix, cls}] = 1.0
          end
      end
    end
  end



  --overlaps = scipy.sparse.csr_matrix(overlaps)

  return {boxes = boxes,
    gt_classes = gt_classes,
    gt_overlaps = overlaps,
    flipped = false}
end

function pascal_voc:get_comp_id(self)
  local comp_id
  if self.config.use_salt then
    comp_id = string.format("%s_%s",self.comp_id, self.salt)
  else
    comp_id = string.format("%s_%s",self.comp_id, self.comp_id)
  end
  return comp_id
end

function create_training_data(dataset_name, pascal_voc, validation_size,output_fn)
  --assert(path.exists(devkit_path),string.format('VOCdevkit path does not exist: %s',devkit_path))
  
  local class_names = pascal_voc.classes --{'__background__', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'}
  local d =pascal_voc
  local image_index_set = d:load_image_set_index()
  local ground_truth = {}
  for i = 1,#d.roidb do
    for j = 1,d.roidb[i].gt_classes:size()[1] do
      if d.roidb[i].gt_classes[j] ~= 0 then
          local box =d.roidb[i].boxes[j]
          local roi = {
            rect = Rect.new(box[1], box[2], box[3], box[4]),
            class_index = d.roidb[i].gt_classes[j],
            class_name = d.classes[d.roidb[i].gt_classes[j]]
          }
                -- generate path relative to annotation dir and join with data dir
          local image_path = d:image_path_from_index(image_index_set[i])
          local file_entry = ground_truth[image_path]
          if not file_entry then
            file_entry = { image_file_name = image_path, rois = {} }
            ground_truth[image_path] = file_entry
          end 
          table.insert(file_entry.rois, roi)
      end
    end
  end
  
  local class_index = pascal_voc.class_index
  local file_names = keys(ground_truth)
  validation_size = validation_size or 0.2 -- 80:20 split
  if validation_size >= 0 and validation_size < 1 then
    validation_size = math.ceil(#file_names * validation_size)
  end
  shuffle(file_names)
  local validation_set = remove_tail(file_names, validation_size)
  local training_set = file_names
  

  print(string.format('Total images: %d; classes: %d; train_set: %d; validation_set: %d; ', 
    #file_names, #class_names, #training_set, #validation_set))
  save_obj(
    output_fn,
    {
      dataset_name = dataset_name,
      ground_truth = ground_truth,
      training_set = training_set,
      validation_set = validation_set,
      class_names = class_names,
      class_index = class_index,
      background_files = ""
    }
  )
  print('Done.')
end  

print("prepare pascal voc 2007")
d = Pascal_voc('trainval', 2007)
--print(d.roidb)

print("create training data:")
create_training_data('trainval', d ,0.2, 'pascal_voc_2007.t7')
