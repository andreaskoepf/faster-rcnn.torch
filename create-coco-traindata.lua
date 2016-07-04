require 'utilities'
require 'Rect'

local json = require 'dkjson'


function read_instances(fn)
  local f = io.open(fn)
  local str = f:read('*all')
  f:close()
  local m = json.decode(str)

  print('#images:')
  print(#m.images)
  print('#annotations:')
  print(#m.annotations)

  local filemap = {}
  local class_names = {}
  local class_index = {}

  for j=1,#m.images do
    local image_file_name = m.images[j].file_name
    for i=1,#m.annotations do
      if m.annotations[i].image_id == m.images[j].id then

        -- Here the indexing of class names might seem a little bit weird, 
        -- since the class names are numbers... (but this works quite well)
        local class_name = string.format("%d", m.annotations[i].category_id)
        if not class_index[class_name] then
          class_names[#class_names + 1] = class_name
          class_index[class_name] = #class_names 
        end 

        -- Order of rectangle corners in Rect.lua: minX, minY, maxX, maxY (in coco annotations: x, y, width, height)
        local roi_entry = {
          rect = Rect.new(m.annotations[i].bbox[1], m.annotations[i].bbox[2], m.annotations[i].bbox[1]+m.annotations[i].bbox[3], m.annotations[i].bbox[2]+m.annotations[i].bbox[4]),
          class_name = class_name,
          class_index = class_index[class_name]
        }
 
        local file_entry = filemap[image_file_name]
        if file_entry == nil then
          file_entry = { image_file_name = image_file_name, rois = {} }
          filemap[image_file_name] = file_entry
        end

        table.insert(file_entry.rois, roi_entry)

      end
    end
  end

  return filemap, class_names, class_index
end


function create_training_data(dataset_name, instances_fn, background_dir, output_fn, validation_size)
  local ground_truth, class_names, class_index = read_instances(instances_fn)
  local file_names = keys(ground_truth)

  validation_size = validation_size or 0.2 -- 80:20 split
  if validation_size >= 0 and validation_size < 1 then
    validation_size = math.ceil(#file_names * validation_size)
  end
  shuffle(file_names)
  local validation_set = remove_tail(file_names, validation_size)
  local training_set = file_names

  local background_files = {}
  if  background_dir then
    background_files = list_files(background_dir, nil, false)
  end

  print(string.format('Total images containing bboxes: %d; classes: %d; train_set: %d; validation_set: %d; background: %d', 
    #keys(ground_truth), #class_names, #training_set, #validation_set, #background_files))

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


create_training_data('mscoco', '/data/coco/annotations/instances_val2014.json', nil, 'mscoco.t7')
--create_training_data('mscoco', '/data/coco/annotations/instances_train2014.json', nil, 'mscoco.t7')

-- Unfortunately, using 'instances_train2014.json' (371 MB) gives the following error:
-- Getting "PANIC: unprotected error in call to Lua API (not enough memory)"
-- A solution to this problem seems to be the usage of lua (version 5.2) instead of luaJIT.
-- See the following discussion: https://github.com/soumith/mscoco.torch/issues/1
-- Instructions for installing torch with Lua 5.2 instead of LuaJIT can be found at: http://torch.ch/docs/getting-started.html
