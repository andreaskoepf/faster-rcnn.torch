local Rect = torch.class('Rect')

-- Rect - helps to deal with screen space coordinates.

-- A rect is defined by the two points (minx, miny) and (maxx, maxy). 
-- On a computer screen with a coordinate system starting in the left
-- upper corner minx, miny, maxx, maxy represent left, top, right and
-- bottom. When dealing with integer pixel coordinates [minx, maxx) 
-- and [miny, maxy) are considered to be half-closed intervals, the  
-- point (maxx, maxy) lies outside the rect.

function Rect:__init(minx, miny, maxx, maxy)
  if type(minx) == 'table' then
    self.minx = minx.minx
    self.miny = minx.miny
    self.maxx = minx.maxx
    self.maxy = minx.maxy
  else
    self.minx = minx
    self.miny = miny
    self.maxx = maxx
    self.maxy = maxy
  end
end

function Rect.empty()
  return Rect.new(0, 0, 0, 0)
end

function Rect.fromXYWidthHeight(x, y, width, height)
  return Rect.new(x, y, x + width, y + height)
end

function Rect.fromCenterWidthHeight(centerX, centerY, width, height)
  return Rect.fromXYWidthHeight(centerX - width * 0.5, centerY - height * 0.5, width, height)
end

function Rect:scale(factorX, factorY)
  if factorY == nil then
    factorY = factorX
  end
  return Rect.new(self.minx * factorX, self.miny * factorY, self.maxx * factorX, self.maxy * factorY)
end

function Rect:inflate(x, y)
  return Rect.new(self.minx - x, self.miny - y, self.maxx + x, self.maxy + y)
end

function Rect:size()
  return self:width(), self:height()
end

function Rect:width()
  return self.maxx-self.minx
end

function Rect:height()
  return self.maxy-self.miny
end

function Rect:area()
  return self:width() * self:height()
end

function Rect:center()
  return (self.minx + self.maxx) / 2, (self.miny + self.maxy) / 2
end

function Rect:isEmpty()
  return self.minx == self.maxx and self.miny == self.maxy
end

function Rect:clip(clipRect)
  return Rect.new(
    math.min(math.max(self.minx, clipRect.minx), clipRect.maxx),
    math.min(math.max(self.miny, clipRect.miny), clipRect.maxy),
    math.max(math.min(self.maxx, clipRect.maxx), clipRect.minx),
    math.max(math.min(self.maxy, clipRect.maxy), clipRect.miny)
  )
end

function Rect:containsPt(x, y)
  return self.minx <= x and x < self.maxx and self.miny <= y and y < self.maxy
end

function Rect:contains(otherRect)
  return self:containsPt(otherRect.minx, otherRect.miny) and self:containsPt(otherRect.maxx, otherRect.maxy)
end

function Rect:normalize()
  local l, t, r, b
  if self.minx <= self.maxx then
    l = self.minx
    r = self.maxx
  else 
    l = self.maxx
    r = self.minx
  end
  if self.miny <= self.maxy then
    t = self.miny
    b = self.maxy
  else 
    t = self.maxy
    b = self.miny
  end
  return Rect.new(l, t, r, b)
end

function Rect:unpack()
  return self.minx, self.miny, self.maxx, self.maxy
end

function Rect.union(a, b)
  local minx = math.min(a.minx, b.minx)
  local miny = math.min(a.miny, b.miny)
  local maxx = math.max(a.maxx, b.maxx)
  local maxy = math.max(a.maxy, b.maxy)
  return Rect.new(minx, miny, maxx, maxy)
end

function Rect.intersect(a, b)
  local minx = math.max(a.minx, b.minx)
  local miny = math.max(a.miny, b.miny)
  local maxx = math.min(a.maxx, b.maxx)
  local maxy = math.min(a.maxy, b.maxy)
  if maxx >= minx and maxy >= miny then
    return Rect.new(minx, miny, maxx, maxy)
  else
    return Rect.empty()
  end
end

function Rect.IoU(a, b)
  local i = Rect.intersect(a, b):area() 
  return i / (a:area() + b:area() - i)
end

function Rect:totensor()
  return torch.Tensor({ self.minx, self.miny, self.maxx, self.maxy })
end

function Rect:snapToInt()
  return Rect.new(math.floor(self.minx), math.floor(self.miny), math.ceil(self.maxx), math.ceil(self.maxy))
end

function Rect:offset(x, y)
  return Rect.new(self.minx + x, self.miny + y, self.maxx + x, self.maxy + y)
end

-- returns vertices in clockwise order
function Rect:vertices()
  return torch.Tensor({
    { self.minx, self.miny },
    { self.maxx, self.miny },
    { self.maxx, self.maxy },
    { self.minx, self.maxy }
  })
end

function Rect:__tostring()
  return string.format("{ min: (%f, %f), max: (%f, %f) }", self.minx, self.miny, self.maxx, self.maxy)
end
