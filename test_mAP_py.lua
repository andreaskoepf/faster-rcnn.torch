require 'torch'
local optim = require 'optim'

local function averagePrecision(tp,fp,npos)

  -- compute precision/recall
  fp=torch.cumsum(fp)
  tp=torch.cumsum(tp)
  local rec=tp/npos
  local prec=torch.cdiv(tp, (fp+tp))

  -- compute average precision
  local ap= torch.zeros(1)
  for t=0,1,0.1 do
    local p=torch.max(prec[rec:ge(t)])
    if p < 1 then
      p=0
    end
    ap=ap+p/11
  end
  return rec,prec,ap
end


local function xVOCap(rec,prec)
  -- From the PASCAL VOC 2011 devkit
  local mrec=torch.cat(torch.zeros(1), rec):cat(torch.ones(1))
  local mpre=torch.cat(torch.zeros(1), prec):cat(torch.zeros(1))

  for i=mpre:size(1)-1,1,-1 do
    mpre[i]=math.max(mpre[i],mpre[i+1])
  end

  local indexA = torch.ByteTensor(mrec:size()):zero()
  local indexB = torch.ByteTensor(mrec:size()):zero()
  for ii = 2,mrec:size(1) do
    if mrec[ii-1]~=mrec[ii] then
      indexA[ii] = 1
      indexB[ii-1] = 1
    end
  end
  
  local ap=torch.sum((mrec[indexA]-mrec[indexB]):cmul(mpre[indexB]))
  
  return ap
end



function test()
  local tp = torch.range(1,6)
  local fp = torch.range(1,6)*2
  local npos = 10
  local rec,prec,ap = averagePrecision(tp,fp,npos)

  ap = xVOCap(rec,prec)
  print(ap)
  
end

test()
