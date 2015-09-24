local Rect = torch.class('Rect')

-- Rect - helps to deal with screen space coordinates.

-- A rect is defined by the two points (minx, miny) and (maxx, maxy). 
-- On a computer screen with a coordinate system starting in the left
-- upper corner minx, miny, maxx, maxy represent left, top, right and
-- bottom. When dealing with integer pixel coordinates [minx, maxx) 
-- and [miny, maxy) are considered to be half-closed intervals, the  
-- point (maxx, maxy) lies outside the rect.

function Rect:__init(minX, minY, maxX, maxY)
  if type(minX) == 'table' then
    self.minX = minx.minX
    self.minY = minx.minY
    self.maxX = minx.maxX
    self.maxY = minx.maxY
  else
    self.minX = minX
    self.minY = minY
    self.maxX = maxX
    self.maxY = maxY
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
  return Rect.new(self.minX * factorX, self.minY * factorY, self.maxX * factorX, self.maxY * factorY)
end

function Rect:inflate(x, y)
  return Rect.new(self.minX - x, self.minY - y, self.maxX + x, self.maxY + y)
end

function Rect:size()
  return self:width(), self:height()
end

function Rect:width()
  return self.maxX-self.minX
end

function Rect:height()
  return self.maxY-self.minY
end

function Rect:area()
  return self:width() * self:height()
end

function Rect:center()
  return (self.minX + self.maxX) / 2, (self.minY + self.maxY) / 2
end

function Rect:isEmpty()
  return self.minX == self.maxX and self.minY == self.maxY
end

function Rect:clip(clipRect)
  return Rect.new(
    math.min(math.max(self.minX, clipRect.minX), clipRect.maxX),
    math.min(math.max(self.minY, clipRect.minY), clipRect.maxY),
    math.max(math.min(self.maxX, clipRect.maxX), clipRect.minX),
    math.max(math.min(self.maxY, clipRect.maxY), clipRect.minY)
  )
end

function Rect:containsPt(x, y)
  return self.minX <= x and x < self.maxX and self.minY <= y and y < self.maxY
end

function Rect:contains(otherRect)
  return self:containsPt(otherRect.minX, otherRect.minY) and self:containsPt(otherRect.maxX, otherRect.maxY)
end

function Rect:overlaps(other)
 return self.minX < other.maxX and self.maxX > other.minX 
  and self.minY < other.maxY and self.maxY > other.minY
end

function Rect:normalize()
  local l, t, r, b
  if self.minX <= self.maxX then
    l = self.minX
    r = self.maxX
  else 
    l = self.maxX
    r = self.minX
  end
  if self.minY <= self.maxY then
    t = self.minY
    b = self.maxY
  else 
    t = self.maxY
    b = self.minY
  end
  return Rect.new(l, t, r, b)
end

function Rect:unpack()
  return self.minX, self.minY, self.maxX, self.maxY
end

function Rect.union(a, b)
  local minx = math.min(a.minX, b.minX)
  local miny = math.min(a.minY, b.minY)
  local maxx = math.max(a.maxX, b.maxX)
  local maxy = math.max(a.maxY, b.maxY)
  return Rect.new(minx, miny, maxx, maxy)
end

function Rect.intersect(a, b)
  local minx = math.max(a.minX, b.minX)
  local miny = math.max(a.minY, b.minY)
  local maxx = math.min(a.maxX, b.maxX)
  local maxy = math.min(a.maxY, b.maxY)
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
  return torch.Tensor({ self.minX, self.minY, self.maxX, self.maxY })
end

function Rect:snapToInt()
  return Rect.new(math.floor(self.minX), math.floor(self.minY), math.ceil(self.maxX), math.ceil(self.maxY))
end

function Rect:offset(x, y)
  return Rect.new(self.minX + x, self.minY + y, self.maxX + x, self.maxY + y)
end

-- returns vertices in clockwise order
function Rect:vertices()
  return torch.Tensor({
    { self.minX, self.minY },
    { self.maxX, self.minY },
    { self.maxX, self.maxY },
    { self.minX, self.maxY }
  })
end

function Rect:clone()
  return Rect.new(self)
end

function Rect:__tostring()
  return string.format("{ min: (%.2f, %.2f), max: (%.2f, %.2f), size: (%.2f x %.2f) }", self.minX, self.minY, self.maxX, self.maxY, self:width(), self:height())
end
