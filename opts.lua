--
--  Copyright 2016, JudeLee
--


local M = {}

function M.parse(arg)
	local cmd = torch.CmdLine()
	cmd:addTime()

	cmd:text()
	cmd:text('Training a convnet for region proposals')
	cmd:text()

	cmd:text('=== Select mode (Train/Test) ===')
	cmd:option('-mode', 'train', 'Select train or test mode (default train)')

	cmd:text('=== Training ===')
	cmd:option('-cfg', '', 'configuration file')
	cmd:option('-model', '', 'model factory file path')
	cmd:option('-name', '', 'experiment name, snapshot prefix') 
	cmd:option('-train', '', 'training data file name')
	cmd:option('-snapshot', 1000, 'snapshot interval')
	cmd:option('-plot', 100, 'plot training progress interval')
	cmd:option('-lr', 1E-4, 'learn rate')
	cmd:option('-rms_decay', 0.9, 'RMSprop moving average dissolving factor')
	cmd:option('-opti', 'rmsprop', 'Optimizer')

	cmd:text('=== Testing ===')
	cmd:option('-restore', '', 'network snapshot file name to load')

	cmd:text('=== Misc ===')
	cmd:option('-threads', 2, 'number of threads (default 2)')
	cmd:option('-gpuid', 0, 'device ID (CUDA), (use -1 for CPU)')
	cmd:option('-seed', 0, 'random seed (0 = no fixed seed)')

	local opt = cmd:parse(arg or {})

	return opt
end

return M

