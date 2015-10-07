-- specify the base path of the ILSVRC2015 dataset: 
ILSVRC2015_BASE_DIR = '/data/imagenet/ILSVRC2015/'

require 'lfs'
require 'LuaXML'      -- if missing use luarocks install LuaXML
require 'utilities'
require 'Rect' 

local ground_truth = {}
local class_names = {}
local class_index = {}

function import_file(anno_base, data_base, fn, name_table)
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
      
      -- generate path relative to annotation dir and join with data dir
      local image_path = path.join(data_base, path.relpath(fn, anno_base))  
      
      -- replace 'xml' file ending with 'JPEG'
      image_path = string.sub(image_path, 1, #image_path - 3) .. 'JPEG'    
      
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

function import_directory(anno_base, data_base, directory_path, recursive, name_table)
   for fn in lfs.dir(directory_path) do
    local full_fn = path.join(directory_path, fn)
    local mode = lfs.attributes(full_fn, 'mode') 
    if recursive and mode == 'directory' and fn ~= '.' and fn ~= '..' then
      import_directory(anno_base, data_base, full_fn, true, name_table)
      collectgarbage()
    elseif mode == 'file' and string.sub(fn, -4):lower() == '.xml' then
      import_file(anno_base, data_base, full_fn, name_table)
    end
    if #ground_truth > 10 then
      return
    end
  end
  return l
end

-- recursively search through training and validation directories and import all xml files
function create_ground_truth_file(dataset_name, base_dir, train_annotation_dir, val_annotation_dir, train_data_dir, val_data_dir, background_dirs, output_fn)
  function expand(p)
    return path.join(base_dir, p)
  end
  
  local training_set = {}
  local validation_set = {}
  import_directory(expand(train_annotation_dir), expand(train_data_dir), expand(train_annotation_dir), true, training_set)
  import_directory(expand(val_annotation_dir), expand(val_data_dir), expand(val_annotation_dir), true, validation_set)
  local file_names = keys(ground_truth)
  
  -- compile list of background images
  local background_files = {}
  for i,directory_path in ipairs(background_dirs) do
    directory_path = expand(directory_path)
    for fn in lfs.dir(directory_path) do
      local full_fn = path.join(directory_path, fn)
      local mode = lfs.attributes(full_fn, 'mode')
      if mode == 'file' and string.sub(fn, -5):lower() == '.jpeg' then
        table.insert(background_files, full_fn)
      end
    end
  end
  
  print(string.format('Total images: %d; classes: %d; train_set: %d; validation_set: %d; (Background: %d)', 
    #file_names, #class_names, #training_set, #validation_set, #background_files
  ))
  save_obj(
    output_fn,
    {
      dataset_name = dataset_name,
      ground_truth = ground_truth,
      training_set = training_set,
      validation_set = validation_set,
      class_names = class_names,
      class_index = class_index,
      background_files = background_files
    }
  )
  print('Done.')
end


background_folders = {}
for i=0,10 do
  table.insert(background_folders, 'Data/DET/train/ILSVRC2013_train_extra' .. i)
end

create_ground_truth_file(
  'ILSVRC2015_DET',
  ILSVRC2015_BASE_DIR,
  'Annotations/DET/train', 
  'Annotations/DET/val',
  'Data/DET/train',
  'Data/DET/val',
  background_folders,
  'ILSVRC2015_DET.t7'
)
