require 'lfs' -- lua file system for directory listings
require 'nn'

function list_files(directory_path, max_count)
  local l = {}
  for fn in lfs.dir(directory_path) do
    if max_count and #l >= max_count then
      break
    end
    local full_fn = path.join(directory_path, fn)
    if lfs.attributes(full_fn, 'mode') == 'file' then 
      table.insert(l, full_fn)
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

function draw_rectangle(img, rect, color)
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
end

function remove_quotes(s)
  return s:gsub('^"(.*)"$', "%1")
end
