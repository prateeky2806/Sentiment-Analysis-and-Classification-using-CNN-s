require 'torch'
require 'nn'
require 'nngraph'

local CNN_static_non_static = torch.class('CNN_static_non_static')

function CNN_static_non_static:architecture(w2v)
	local input = nn.Identity()()
	-- Table storing weights,gradweights, padding value.
	local lookuptable = nn.LookupTable(opt.vocabSize,opt.vecSize)
	lookuptable.weight:copy(w2v)
	lookuptable.weight[1]:zero()
	lookuptable = lookuptable(input)
	
	local kernels = opt.kernels
	local layer = {}
	for i=1, #kernels do
		local convSetup
		local convLayer
		local maxTimePool
		if opt.cudnn==1 then
			require 'cudnn'
			require 'cunn'
			convSetup = cudnn.SpatialConvolution(1,opt.numFeatMaps,opt.vecSize,kernels[i])
			convLayer = nn.Reshape(opt.numFeatMaps, opt.maxSent-kernels[i]+1, true)(
				     convSetup(nn.Reshape(1, opt.maxSent, opt.vecSize, true)(lookuptable)))
			maxTimePool = nn.Max(3)(cudnn.ReLU()(convLayer))
		else
			convSetup = nn.TemporalConvolution(opt.vecSize, opt.numFeatMaps, kernels[i])
			convLayer = convSetup(lookuptable)
			maxTimePool = nn.Max(2)(nn.ReLU()(convLayer))
		end
		convSetup.weight:uniform(-0.2,0.2)
		convSetup.bias:zero()
		table.insert(layer, maxTimePool)
	end
	local convLayerConcat
	if #layer > 1 then
		convLayerConcat = nn.JoinTable(2)(layer)
	else
		convLayerConcat = layer[1]
	end
	local lastLayer = convLayerConcat
	local linear = nn.Linear((#layer) * opt.numFeatMaps, opt.numClasses)
	linear.weight:normal():mul(0.01)
	linear.bias:zero()

	local softmax
	if opt.cudnn == 1 then
		softmax = cudnn.LogSoftMax()
	else
		softmax = nn.LogSoftMax()
	end

	local output = softmax(linear(nn.Dropout(opt.dropout_p)(lastLayer))) 
	model = nn.gModule({input}, {output})
	return model
end

return CNN_static_non_static
