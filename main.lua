require 'hdf5'
require 'nn'
require 'optim'
require 'lfs'
require 'nngraph'


cmd = torch.CmdLine()

cmd:option('-model_type', 'rand', 'Model type. Options: rand , static , nonstatic)')
cmd:option('-data', 'TREC.hdf5', 'Training data and word2vec data')
cmd:option('-cudnn', 0, '')
cmd:option('-folds', 10, 'number of folds to use. If test set provided, folds=1. max 10')
cmd:text()

-- Hyperparameters (Defaults as mentioned in the paper)
cmd:option('-numEpochs', 25, 'Number of training epochs')
cmd:option('-L2s', 3, 'L2 normalize weights')
cmd:option('-batchSize', 50, 'Batch size for training')
cmd:option('-numFeatMaps', 100, 'Number of feature maps after 1st convolution')
cmd:option('-kernels', '{3,4,5}', 'Kernel sizes of convolutions, table format.')
cmd:option('-dropout_p', 0.5, 'p for dropout')
cmd:text()


function loadData()
  local train, trainLabel
  local dev, devLabel
  local test, testLabel

  print('loading data...')
  local f = hdf5.open(opt.data, 'r')
  local w2v = f:read('w2v'):all()
  train = f:read('train'):all()
  trainLabel = f:read('trainLabel'):all()
  opt.numClasses = torch.max(trainLabel)
# These opt.hasDev, opt.hasTest values are assigned based on the HDF5 file loaded, the value is provided by data file.
  if f:read('dev'):dataspaceSize()[1] == 0 then
    opt.hasDev = 0
  else
    opt.hasDev = 1
    dev = f:read('dev'):all()
    devLabel = f:read('devLabel'):all()
  end
  if f:read('test'):dataspaceSize()[1] == 0 then
    opt.hasTest = 0
  else
    opt.hasTest = 1
    test = f:read('test'):all()
    testLabel = f:read('testLabel'):all()
  end
  print('data loaded!')

  return train, trainLabel, test, testLabel, dev, devLabel, w2v
end


function trainLoop(allTrain, allTrainLabel, test, testLabel, dev, devLabel, w2v)
  -- Initialize objects 
  local Train = require 'train'
  local trainModel = Train.new()
  local Test = require 'test'
  local testModel = Test.new()

  local optimMethod
    optimMethod = optim.adadelta

  local bestModel -- save best model
  local foldDevScores = {}
  local foldTestScores = {}

  local train, trainLabel -- trainModel set for each fold
  if opt.hasTest == 1 then
    train = allTrain
    trainLabel = allTrainLabel
    
  end

  -- Training folds.
  for fold = 1, opt.folds do

    print()
    print('==> fold ', fold)

    if opt.hasTest == 0 then
      -- Spliting code is no test set, refered from internet.
      -- make train/test data (90/10 split for train/test)
      local N = allTrain:size(1)
      local i_start = math.floor((fold - 1) * (N / opt.folds) + 1)
      local i_end = math.floor(fold * (N / opt.folds))
      test = allTrain:narrow(1, i_start, i_end - i_start + 1)
      testLabel = allTrainLabel:narrow(1, i_start, i_end - i_start + 1)
      train = torch.cat(allTrain:narrow(1, 1, i_start), allTrain:narrow(1, i_end, N - i_end + 1), 1)
      trainLabel = torch.cat(allTrainLabel:narrow(1, 1, i_start), allTrainLabel:narrow(1, i_end, N - i_end + 1), 1)
      
    end

    if opt.hasDev == 0 then
      -- Shuffling training data to get dev/train split (10% to dev), refered from internet
      -- We organize our data in batches at this split before epoch training.
      local J = train:size(1)
      local shuffle = torch.randperm(J):long()
      train = train:index(1, shuffle)
      trainLabel = trainLabel:index(1, shuffle)

      local numBatches = math.floor(J / opt.batchSize)
      local numTrainBatches = torch.round(numBatches * 0.9)

      local trainSize = numTrainBatches * opt.batchSize
      local devSize = J - trainSize
      dev = train:narrow(1, trainSize+1, devSize)
      devLabel = trainLabel:narrow(1, trainSize+1, devSize)
      train = train:narrow(1, 1, trainSize)
      trainLabel = trainLabel:narrow(1, 1, trainSize)
    end

    -- build model
    local model, criterion, layers = buildModel(w2v)

    -- Call getParameters once
    local params, grads = model:getParameters()

    -- Training loop.
    bestModel = model:clone()
    local bestEpoch = 1
    local bestErr = 0.0

    -- Training.
    -- Gradient descent state should persist over epochs
    local state = {}
    for epoch = 1, opt.numEpochs do
      -- Train
      local trainErr = trainModel:trainIt(train, trainLabel, model, criterion, optimMethod, layers, state, params, grads)
      -- Dev
      local devErr = testModel:testIt(dev, devLabel, model, criterion)
      if devErr > bestErr then
        bestModel = model:clone()
        bestEpoch = epoch
        bestErr = devErr 
      end

      print('epoch:', epoch, 'train perf:', 100*trainErr, '%, val perf ', 100*devErr, '%')
    end

    print('best dev err:', 100*bestErr, '%, epoch ', bestEpoch)
    table.insert(foldDevScores, bestErr)

    local testErr = testModel:testIt(test, testLabel, bestModel, criterion)
    print('test perf ', 100*testErr, '%')
    table.insert(foldTestScores, testErr)
  end

  return foldDevScores, foldTestScores, bestModel
end

-- Creating model for training according to user specifications
function buildModel(w2v)
  if opt.modelType == 'static' or opt.modelType == 'nonstatic' then
    local CNN_static_non_static = require 'model.CNN_static_non_static'
    local modelBuilder = CNN_static_non_static.new()
    local model
    model = modelBuilder:architecture(w2v)

  else
    local CNN_rand = require 'model.CNN_rand'
    local modelBuilder = CNN_rand.new()
    local model
    model = modelBuilder:architecture(w2v)
  end

  local criterion = nn.ClassNLLCriterion()

  -- Run Code on GPU for much faster training.
  if opt.cudnn == 1 then
    model = model:cuda()
    criterion = criterion:cuda()
  end

  -- 
  local layers = {}
  layers['linear'] = getLayer(model, 'nn.Linear')
  layers['w2v'] = getLayer(model, 'nn.LookupTable')
  
  return model, criterion, layers
end


function getLayer(model, name)
  local namedLayer
  function get(layer)
    if torch.typename(layer) == name or layer.name == name then
      namedLayer = layer
    end
  end

  model:apply(get)
  return namedLayer
end


function main()
  -- parse arguments
  opt = cmd:parse(arg)

  torch.manualSeed(-1)
  if opt.cudnn == 1 then
    require 'cutorch'
    cutorch.manualSeedAll(-1)
    cutorch.setDevice(1)
  end

  -- Read HDF5 training data
  local train, trainLabel
  local test, testLabel
  local dev, devLabel
  local w2v
  train, trainLabel, test, testLabel, dev, devLabel, w2v = loadData()

  opt.vocabSize = w2v:size(1)
  opt.vecSize = w2v:size(2)
  opt.maxSent = train:size(2)
  print('vocab size: ', opt.vocabSize)
  print('vec size: ', opt.vecSize)
  print('max sentence size: ', opt.maxSent)

  -- Retrieve kernels
  loadstring("opt.kernels = " .. opt.kernels)()

  if opt.hasTest == 1 then
    -- No cross validation if we have a test set
    opt.folds = 1
  end

  -- training and printing final scores
  local foldDevScores, foldTestScores, bestModel = trainLoop(train, trainLabel, test, testLabel, dev, devLabel, w2v)

  print('dev scores:')
  print(foldDevScores)
  print('average dev score: ', torch.Tensor(foldDevScores):mean())

  print('test scores:')
  print(foldTestScores)
  print('average test score: ', torch.Tensor(foldTestScores):mean())
  

  local savefile
    savefile = string.format('results/%s_model.t7', os.date('%Y%m%d_%H%M'))
  print('saving results to ', savefile)

  local save = {}
  save['devScores'] = foldDevScores
  save['testScores'] = foldTestScores
  save['opt'] = opt
  save['model'] = bestModel
  save['embeddings'] = getLayer(bestModel, 'nn.LookupTable').weight
  torch.save(savefile, save)
end

main()
