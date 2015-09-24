require 'lfs'
require 'LuaXML'
require 'utilities'
require 'Rect' 

local ground_truth = {}
local class_names = {}
local class_index = {}

function import_file(fn, name_table)
  local x = xml.load(fn)
  local a = x:find('annotation')
  local folder = a:find('folder')[1]
  local filename = a:find('filename')[1]
  local src = a:find('source')
  local db = src:find('database')[1]
  local sz = a:find('size')
  local w = tonumber(sz:find('width')[1])
  local h = tonumber(sz:find('height')[1])
  
  for _,e in pairs(a) do
    if e[e.TAG] == 'object' then
    
      local obj = e
      local name = obj:find('name')[1]
      local bb = obj:find('bndbox') 
      local xmin = tonumber(bb:find('xmin')[1])
      local xmax = tonumber(bb:find('xmax')[1])
      local ymin = tonumber(bb:find('ymin')[1])
      local ymax = tonumber(bb:find('ymax')[1])
      
      if not class_index[name] then
        class_names[#class_names + 1] = name
        class_index[name] = #class_names 
      end 
      
      local image_path = path.join(folder, filename) .. '.JPEG'
      table.insert(name_table, image_path)
      
      local roi = {
        rect = Rect.new(xmin, ymin, xmax, ymax),
        class_index = class_index[name],
        class_name = name
      }
      
      local file_entry = ground_truth[image_path]
      if not file_entry then
        file_entry = { image_file_name = image_path, rois = {} }
        ground_truth[image_path] = file_entry
      end 
      table.insert(file_entry.rois, roi)
    end
  end
end

function import_directory(directory_path, recursive, name_table)
   for fn in lfs.dir(directory_path) do
    local full_fn = path.join(directory_path, fn)
    local mode = lfs.attributes(full_fn, 'mode') 
    if recursive and mode == 'directory' and fn ~= '.' and fn ~= '..' then 
      import_directory(full_fn, true, name_table)
      collectgarbage()
    elseif mode == 'file' and string.sub(fn, -4):lower() == '.xml' then
      import_file(full_fn, name_table)
    end
    if #ground_truth > 10 then
      return
    end
  end
  return l
end

function create_ground_truth_file(dataset_name, train_annotation_dir, valid_annotation_dir, output_fn)
  local training_set = {}
  local validation_set = {}
  import_directory(train_annotation_dir, true, training_set)
  import_directory(valid_annotation_dir, true, validation_set)
  local file_names = keys(ground_truth)
  print(string.format('Total images: %d; classes: %d; train_set: %d; validation_set: %d', #file_names, #class_names, #training_set, #validation_set))
  save_obj(
    output_fn,
    {
      dataset_name = dataset_name,
      ground_truth = ground_truth,
      training_set = training_set,
      validation_set = validation_set,
      class_names = class_names,
      class_index = class_index
    }
  )
  print('Done.')
end

create_ground_truth_file(
  'ILSVRC2015_DET', 
  '/data/imagenet/ILSVRC2015/Annotations/DET/train', 
  '/data/imagenet/ILSVRC2015/Annotations/DET/val',
  'ILSVRC2015_DET.t7'
)
