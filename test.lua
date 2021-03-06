require 'nn'
require 'sys'
require 'torch'

local Test = torch.class('Test')

function Test:testIt(testData, testLabels, model, criterion)
  model:evaluate()
  local classes = {}
  for i = 1, opt.numClasses do
    table.insert(classes, i)
  end
  local confusion = optim.ConfusionMatrix(classes)
  confusion:zero()

  local testSize = testData:size(1)

  local totalErr = 0

  for t = 1, testSize, opt.batchSize do
    -- data samples and labels, in mini batches.
    local batchSize = math.min(opt.batchSize, testSize - t + 1)
    local inputs = testData:narrow(1, t, batchSize)
    local targets = testLabels:narrow(1, t, batchSize)
    if opt.cudnn == 1 then
      inputs = inputs:cuda()
      targets = targets:cuda()
    else
      inputs = inputs:double()
      targets = targets:double()
    end

    local outputs = model:forward(inputs)
    local err = criterion:forward(outputs, targets)
    totalErr = totalErr + err * batchSize

    for i = 1, batchSize do
      confusion:add(outputs[i], targets[i])
    end
  end

  -- return error percent
  confusion:updateValids()
  return confusion.totalValid
end
return Test
