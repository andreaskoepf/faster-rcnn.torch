require 'lfs' -- lua file system for directory listings
require 'nn'
local image = require 'image'

function list_files(directory_path, max_count, abspath)
  local l = {}
  for fn in lfs.dir(directory_path) do
    if max_count and #l >= max_count then
      break
    end
    local full_fn = path.join(directory_path, fn)
    if lfs.attributes(full_fn, 'mode') == 'file' then
      table.insert(l, abspath and full_fn or fn)
    end
  end
  return l
end

function clamp(x, lo, hi)
  return math.max(math.min(x, hi), lo)
end

function saturate(x)
  return clam(x, 0, 1)
end

function lerp(a, b, t)
  return (1-t) * a + t * b
end

function shuffle_n(array, count)
  count = math.max(count, count or #array)
  local r = #array    -- remaining elements to pick from
  local j, t
  for i=1,count do
    j = math.random(r) + i - 1
    t = array[i]    -- swap elements at i and j
    array[i] = array[j]
    array[j] = t
    r = r - 1
  end
end

function shuffle(array)
  local i, t
  for n=#array,2,-1 do
    i = math.random(n)
    t = array[n]
    array[n] = array[i]
    array[i] = t
  end
  return array
end

function shallow_copy(t)
  local t2 = {}
  for k,v in pairs(t) do
    t2[k] = v
  end
  return t2
end

function deep_copy(obj, seen)
  if type(obj) ~= 'table' then
    return obj
  end
  if seen and seen[obj] then
    return seen[obj]
  end
  local s = seen or {}
  local res = setmetatable({}, getmetatable(obj))
  s[obj] = res
  for k, v in pairs(obj) do
    res[deep_copy(k, s)] = deep_copy(v, s)
  end
  return res
end

function reverse(array)
  local n = #array, t
  for i=1,n/2 do
    t = array[i]
    array[i] = array[n-i+1]
    array[n-i+1] = t
  end
  return array
end

function remove_tail(array, num)
  local t = {}
  for i=num,1,-1 do
    t[i] = table.remove(array)
  end
  return t, array
end

function keys(t)
  local l = {}
  for k,v in pairs(t) do
    table.insert(l, k)
  end
  return l
end

function values(t)
  local l = {}
  for k,v in pairs(t) do
    table.insert(l, v)
  end
  return l
end

function save_obj(file_name, obj)
  local f = torch.DiskFile(file_name, 'w')
  f:writeObject(obj)
  f:close()
end

function load_obj(file_name)
  local f = torch.DiskFile(file_name, 'r')
  local obj = f:readObject()
  f:close()
  return obj
end

function save_model(file_name, weights, options, stats)
  save_obj(file_name,
    {
      version = 0,
      weights = weights,
      options = options,
      stats = stats
    })
end

function combine_and_flatten_parameters(...)
  local nets = { ... }
  local parameters,gradParameters = {}, {}
  for i=1,#nets do
    local w, g = nets[i]:parameters()
    for i=1,#w do
      table.insert(parameters, w[i])
      table.insert(gradParameters, g[i])
    end
  end
  return nn.Module.flatten(parameters), nn.Module.flatten(gradParameters)
end

function draw_rectangle(img, rect, color, label)
  label = label  or ""
  local sz = img:size()

  local x0 = math.max(1, rect.minX)
  local x1 = math.min(sz[3], rect.maxX)
  local w = math.floor(x1) - math.floor(x0)
  if w >= 0 then
    local v = color:view(3,1):expand(3, w + 1)
    if rect.minY > 0 and rect.minY <= sz[2] then
      img[{{}, rect.minY, {x0, x1}}] = v
    end
    if rect.maxY > 0 and rect.maxY <= sz[2] then
      img[{{}, rect.maxY, {x0, x1}}] = v
    end
  end

  local y0 = math.max(1, rect.minY)
  local y1 = math.min(sz[2], rect.maxY)
  local h = math.floor(y1) - math.floor(y0)
  if h >= 0 then
    local v = color:view(3,1):expand(3, h + 1)
    if rect.minX > 0 and rect.minX <= sz[3] then
      img[{{}, {y0, y1}, rect.minX}] = v
    end
    if rect.maxX > 0 and rect.maxX <= sz[3] then
      img[{{}, {y0, y1}, rect.maxX}] = v
    end
  end

  if h > 0 and w > 0 and x0>10 and y0 < y1-10 then
    --print(string.format("drawText '%s' at x0 = %f, y0 = %f,info: x1 = %f, y1 = %f, image width = %f, height = %f ",label,x0,y0,x1,y1,sz[3],sz[2]))
    img:copy(image.drawText(img:double(),label,x0,y0,{color = {color[1]*255,color[2]*255,color[3]*255}, size =1}))
    img:cuda()
  end
  return img
end


function remove_quotes(s)
  return s:gsub('^"(.*)"$', "%1")
end

function normalize_debug(t)
  local lb, ub = t:min(), t:max()
  return (t -lb):div(ub-lb+1e-10)
end

function find_target_size(orig_w, orig_h, target_smaller_side, max_pixel_size)
  local w, h
  if orig_h < orig_w then
    -- height is smaller than width, set h to target_size
    w = math.min(orig_w * target_smaller_side/orig_h, max_pixel_size)
    h = math.floor(orig_h * w/orig_w + 0.5)
    w = math.floor(w + 0.5)
  else
    -- width is smaller than height, set w to target_size
    h = math.min(orig_h * target_smaller_side/orig_w, max_pixel_size)
    w = math.floor(orig_w * h/orig_h + 0.5)
    h = math.floor(h + 0.5)
  end
  assert(w >= 1 and h >= 1)
  return w, h
end

function find_scaled_size(orig_w, orig_h, target_smaller_side)
  local w, h
  if orig_h < orig_w then
    -- height is smaller than width, set h to target_size
    h = target_smaller_side
    w = math.floor(orig_w * h/orig_h + 0.5)
  else
    -- width is smaller than height, set w to target_size
    w = target_smaller_side
    h = math.floor(orig_h * w/orig_w + 0.5)
  end
  assert(w >= 1 and h >= 1)
  return w, h
end

function load_image(fn, color_space, base_path)
  if not path.isabs(fn) and base_path then
    fn = path.join(base_path, fn)
  end
  local img = image.load(fn, 3, 'float')
  if color_space == 'yuv' then
    img = image.rgb2yuv(img)
  elseif color_space == 'lab' then
    img = image.rgb2lab(img)
  elseif color_space == 'hsv' then
    img = image.rgb2hsv(img)
  end
  return img
end

function printf(...)
  print(string.format(...))
end
