require 'lfs'
require 'LuaXML'
require 'utilities'
require 'Rect' 


function read_csv_file(fn)
  -- format of RoI file:
  -- filename, left, top, right, bottom, model_class_name, model_class_index, material_name, material_index
  -- "img8494058054b911e5a5ab086266c6c775.png", 0, 573, 5a9, 701, "DuploBrick_2x2", 2, "DuploBrightGreen", 11

  local f = io.open(fn, 'r')

  local filemap = {}
  local class_names = {}
  local class_index = {}
  
  for l in f:lines() do
    local v = l:split(',') -- get values of single row (we have a trivial csv file without ',' in string values)
    
    local class_name = remove_quotes(v[6])
    if not class_index[class_name] then
      class_names[#class_names + 1] = class_name
      class_index[class_name] = #class_names 
    end 
      
    local image_file_name = remove_quotes(v[1])  -- extract image file name, remove quotes
    local roi_entry = {
      rect = Rect.new(tonumber(v[2]), tonumber(v[3]), tonumber(v[4]), tonumber(v[5])),
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
 
  f:close()
  
  return filemap, class_names, class_index
end


function create_training_data(dataset_name, csv_file_name, background_dir, output_fn, validation_size)
  local ground_truth, class_names, class_index = read_csv_file(csv_file_name)
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

  print(string.format('Total images: %d; classes: %d; train_set: %d; validation_set: %d; background: %d', 
    #file_names, #class_names, #training_set, #validation_set, #background_files))
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
  
create_training_data('duplo-bricks', '/home/koepf/datasets/brickset_all/boxes.csv', '/home/koepf/datasets/background', 'duplo.t7')
