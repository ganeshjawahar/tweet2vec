--[[

Implementation of the ECIR 2017 paper: 'Improving Tweet Representations using Temporal and User Context'

]]--

require 'torch'
require 'io'
require 'nn'
require 'sys'
require 'optim'
require 'os'
require 'xlua'
require 'lfs'
require 'cunn'
require 'cutorch'
require 'nnx'
require 'cunnx'
tds = require('tds')
paths.dofile('model.lua')
local utils=require 'utils'
HSMClass = require 'HSMClass'
require 'HLogSoftMax'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Improving Tweet Representations using Temporal and User Context')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/spouse/','Directory for accessing the user profile prediction data.')
cmd:option('-glove_dir','data/','Directory for accesssing the pre-trained glove word embeddings')
cmd:option('-pred_dir','predictions/','Prediction or output Directory')
cmd:option('-to_lower',1,'change the case of word to lower case')
-- model params (general)
cmd:option('-wdim',200,'dimensionality of word embeddings')
cmd:option('-wwin',21,'defines context words in a document for word modeling')
cmd:option('-twin',21,'defines context tweets in a stream for tweet modeling')
cmd:option('-min_freq',5,'words that occur less than <int> times will not be taken for training')
cmd:option('-pad_tweet',1,'should we need to pad the tweet ?')
cmd:option('-is_word_center_target',0,'model center element based on its surrounding words?')
cmd:option('-is_tweet_center_target',1,'model center element based on its surrounding tweets?')
cmd:option('-pre_train',1,'initialize word embeddings with pre-trained vectors?')
cmd:option('-wc_mode',2,'word context mode 1-concatenate 2-sum 3-average 4-weighted sum')
cmd:option('-tc_mode',4,'tweet context mode 1-concatenate 2-sum 3-average 4-weighted sum')
cmd:option('-tweet',1,'use tweet based model too?')
cmd:option('-user', 0,'model user?')
cmd:option('-wpred',2,'word model final prediction 1=normal softmax 2=hierarchical softmax 3=brown softmax')
cmd:option('-tpred',2,'tweet model final prediction 1=normal softmax 2=hierarchical softmax 3=brown softmax')
-- optimization
cmd:option('-learning_rate',0.001,'learning rate')
cmd:option('-batch_size',128,'number of sequences to train on in parallel')
cmd:option('-max_epochs',25,'number of full passes through the training data')

-- parse input params
params=cmd:parse(arg)
if params.print_params==1 then
	-- output the parameters	
	for param, value in pairs(params) do
	    print(param ..' : '.. tostring(value))
	end
end
lfs.mkdir(params.pred_dir)
params.train_file=params.data_dir..'/train.txt'
params.dev_file=params.data_dir..'/dev.txt'
params.test_file=params.data_dir..'/test.txt'

params.vocab=tds.hash()
params.index2word=tds.hash()
params.word2index=tds.hash()
params.user2id=tds.hash()
params.id2user=tds.hash()
params.index2tweettext=tds.hash()

-- Build vocabulary
utils.buildVocab(params)

-- Load train, dev, test set into memory
params.train_set = utils.loadTensorsToMemory(params, params.train_file)
params.dev_set = utils.loadTensorsToMemory(params, params.dev_file)
params.test_set = utils.loadTensorsToMemory(params, params.test_file)

-- build the net
if params.wpred == 1 then
	params.word_model = build_model_multi_normal(params.vocab_size,params.wdim,'word_lookup',#params.index2tweettext,params.wdim,'tweet_lookup',params.wdim, params.wwin-1, params.wc_mode, params.vocab_size)
	params.word_criterion = nn.CrossEntropyCriterion()
elseif params.wpred == 2 then
	local tree1, root1 = utils.create_frequency_tree(utils.create_word_map(params.vocab, params.index2word))
	params.word_model = build_model_multi(params.vocab_size,params.wdim,'word_lookup',#params.index2tweettext,params.wdim,'tweet_lookup',params.wdim, params.wwin-1, tree1, root1, params.wc_mode)
	params.word_criterion = nn.TreeNLLCriterion()
elseif params.wpred == 3 then
	params.word_model = build_model_multi_normal(params.vocab_size,params.wdim,'word_lookup',#params.index2tweettext,params.wdim,'tweet_lookup',params.wdim, params.wwin-1, params.wc_mode, params.vocab_size)
	params.word_criterion =  nn.HLogSoftMax(utils.getBrownMapping(params.vocab_size), params.vocab_size)
end
params.word_model = params.word_model:cuda()
params.word_criterion = params.word_criterion:cuda()
params.word_lookup = utils.getLookup(params.word_model, 'word_lookup')
params.tweet_lookup = utils.getLookup(params.word_model, 'tweet_lookup')
if params.tweet == 1 then	
	if params.user == 0 then
		if params.tpred == 1 then
			params.tweet_model = build_model_normal(#params.index2tweettext,params.wdim,'tweet_lookup',params.wdim, params.twin-1, params.tc_mode, #params.index2tweettext)
			params.tweet_criterion = nn.CrossEntropyCriterion()
		elseif params.tpred == 2 then
			local tree2, root2 = utils.create_frequency_tree(utils.create_tweet_map(#params.index2tweettext))
			params.tweet_model = build_model(#params.index2tweettext,params.wdim,'tweet_lookup',params.wdim, params.twin-1, tree2, root2, params.tc_mode)
			params.tweet_criterion = nn.TreeNLLCriterion()
		elseif params.tpred == 3 then
			params.tweet_model = build_model_normal(#params.index2tweettext,params.wdim,'tweet_lookup',params.wdim, params.twin-1, params.tc_mode, #params.index2tweettext)
			params.tweet_criterion =  nn.HLogSoftMax(utils.getBrownMapping(#params.index2tweettext), #params.index2tweettext)
		end
	elseif params.user == 1 then
		if params.tpred == 1 then
			params.tweet_model = build_model_multi_normal(#params.index2tweettext,params.wdim,'tweet_lookup',#params.id2user,params.wdim,'user_lookup',params.wdim, params.twin-1, params.tc_mode, #params.index2tweettext)
			params.tweet_criterion = nn.CrossEntropyCriterion()
		elseif params.tpred == 2 then
			local tree2, root2 = utils.create_frequency_tree(utils.create_tweet_map(#params.index2tweettext))
			params.tweet_model = build_model_multi(#params.index2tweettext,params.wdim,'tweet_lookup',#params.id2user,params.wdim,'user_lookup',params.wdim, params.twin-1, tree2, root2, params.tc_mode)
			params.tweet_criterion = nn.TreeNLLCriterion()
		elseif params.tpred == 3 then
			params.tweet_model = build_model_multi_normal(#params.index2tweettext,params.wdim,'tweet_lookup',#params.id2user,params.wdim,'user_lookup',params.wdim, params.twin-1, params.tc_mode, #params.index2tweettext)
			params.tweet_criterion =  nn.HLogSoftMax(utils.getBrownMapping(#params.index2tweettext), #params.index2tweettext)
		end
	end

	params.tweet_model = params.tweet_model:cuda()
	params.tweet_criterion = params.tweet_criterion:cuda()	

	-- sharing the parameters
	local tlook = utils.getLookup(params.tweet_model, 'tweet_lookup')
	tlook:share(params.tweet_lookup,'weight','bias','gradWeight','gradBias')
end

-- Initialize word vectors with pre-trained word embeddings
if params.pre_train==1 then
	utils.initWordWeights(params,params.glove_dir..'glove.twitter.27B.'..params.wdim..'d.txt.gz')
end

params.cur_tweet_user_tensors=torch.CudaTensor(params.batch_size,1)
params.cur_tweet_target_tensors=torch.CudaTensor(params.batch_size)
params.cur_tweet_context_tensors=torch.CudaTensor(params.batch_size,params.twin-1)

params.cur_tweet_tensor=torch.CudaTensor(params.batch_size,1)
params.cur_word_target_tensors=torch.CudaTensor(params.batch_size)
params.cur_word_context_tensors=torch.CudaTensor(params.batch_size,params.wwin-1)

params.w_optim_state={learningRate=params.learning_rate}
params.w_params,params.w_grad_params=params.word_model:getParameters()
params.w_params:normal(0, 0.05)
if params.wpred == 3 then 
	params.word_hsm_params, params.word_hsm_grad_params = params.word_criterion:getParameters() 
	params.word_hsm_params:normal(0, 0.05)
end
params.w_feval=function(x)
	if x ~= params.w_params then
		params.w_params:copy(x)
	end
	if params.wpred == 3 then
		params.word_hsm_grad_params:zero()
	end
	params.w_grad_params:zero()
	local inp = {params.cur_word_context_tensors, params.cur_tweet_tensor}
	if params.wpred == 2 then
		table.insert(inp, params.cur_word_target_tensors)
	end

	local out = params.word_model:forward(inp)
	local loss = params.word_criterion:forward(out, params.cur_word_target_tensors)
	local grads = params.word_criterion:backward(out, params.cur_word_target_tensors)
	params.word_model:backward(inp, grads)
	-- params.w_grad_params:div(params.batch_size)
	if params.wpred == 3 then
		params.word_hsm_params:add(params.word_hsm_params:mul(-1 * params.w_optim_state.learningRate))
	end
	return (loss/params.batch_size), params.w_grad_params
end
if params.tweet == 1 then	
	params.t_optim_state={learningRate=params.learning_rate}
	params.t_params,params.t_grad_params=params.tweet_model:getParameters()
	params.t_params:normal(0, 0.05)
	if params.tpred == 3 then 
		params.tweet_hsm_params, params.tweet_hsm_grad_params = params.tweet_criterion:getParameters() 
		params.tweet_hsm_params:normal(0, 0.05)
	end
	params.t_feval=function(x)
		if x ~= params.t_params then
			params.t_params:copy(x)
		end
		if params.tpred == 3 then
			params.tweet_hsm_grad_params:zero()
		end
		params.t_grad_params:zero()
		local inp = {params.cur_tweet_context_tensors}
		if params.user == 1 then table.insert(inp, params.cur_tweet_user_tensors) end
		if params.tpred == 2 then table.insert(inp, params.cur_tweet_target_tensors) end
		if #inp == 1 then inp = unpack(inp) end
		local out = params.tweet_model:forward(inp)
		local loss = params.tweet_criterion:forward(out, params.cur_tweet_target_tensors)
		local grads = params.tweet_criterion:backward(out, params.cur_tweet_target_tensors)
		params.tweet_model:backward(inp, grads)
		-- params.t_grad_params:div(params.batch_size)
		if params.tpred == 3 then
			params.tweet_hsm_params:add(params.tweet_hsm_params:mul(-1 * params.t_optim_state.learningRate))
		end
		return (loss/params.batch_size), params.t_grad_params
	end
end

function run_model(dataset)
	for epoch=1,params.max_epochs do
		local epoch_start=sys.clock()
		local w_epoch_loss=0
		local w_epoch_iteration=0
		local t_epoch_loss=0
		local t_epoch_iteration=0

		-- Modeling word likelihood
		local indices=torch.randperm(dataset.wm_size)
		print('Modeling word likelihood...')
		for index=1,dataset.wm_size,params.batch_size do
			xlua.progress(index,dataset.wm_size)
			local last_sample=math.min(index+params.batch_size-1,dataset.wm_size)
			local j = 0
			for i=index,last_sample do
				i = indices[i]
				j = j + 1
				params.cur_word_context_tensors[j] = dataset.word_model_word_context[i]
				params.cur_word_target_tensors[j] = dataset.word_model_word_target[i]
				params.cur_tweet_tensor[j] = dataset.word_model_tweet_target[i]
			end
			local _,loss=optim.adam(params.w_feval,params.w_params,params.w_optim_state)
			w_epoch_loss=w_epoch_loss+loss[1]
			w_epoch_iteration=w_epoch_iteration+1
		end
		xlua.progress(dataset.wm_size,dataset.wm_size)

		if params.tweet==1 then
		-- Modeling tweet likelihood
		local indices=torch.randperm(dataset.tm_size)
		print('Modeling tweet likelihood...')
		for index=1,dataset.tm_size,params.batch_size do
			xlua.progress(index,dataset.tm_size)
			local last_sample=math.min(index+params.batch_size-1,dataset.tm_size)
			local j = 0
			for i=index,last_sample do
				i = indices[i]
				j = j + 1
				params.cur_tweet_context_tensors[j] = dataset.tweet_model_context[i]
				params.cur_tweet_target_tensors[j] = dataset.tweet_model_target[i]
				params.cur_tweet_user_tensors[j] = dataset.tweet_model_user_context[i]
			end
			local _,loss=optim.adam(params.t_feval,params.t_params,params.t_optim_state)
			t_epoch_loss=t_epoch_loss+loss[1]
			t_epoch_iteration=t_epoch_iteration+1
		end
		xlua.progress(dataset.tm_size,dataset.tm_size)
		end
		print(string.format("Epoch %d done in %.2f minutes. word loss=%f tweet loss=%f\n\n",epoch,((sys.clock()-epoch_start)/60),(w_epoch_loss/w_epoch_iteration),(t_epoch_loss/t_epoch_iteration)))
	end
end

-- Training
print('Training...')
local start=sys.clock()
run_model(params.train_set)
print(string.format("Training done in %.2f minutes.",((sys.clock()-start)/60)))

-- Evaluation
function compute_vectors(dataset, out)
	params.w_optim_state={learningRate=params.learning_rate}
	params.t_optim_state={learningRate=params.learning_rate}
	run_model(dataset)
	utils.saveTweetEmbeddings(params.tweet_lookup.weight,out,dataset.key_list,dataset.entity_map)
end

params.word_lookup.accGradParameters=function() end
utils.saveEmbeddings('word',params.word_lookup.weight,params.index2word,params.pred_dir..'word_feat.txt')
local start=sys.clock()
utils.saveTweetEmbeddings(params.tweet_lookup.weight,params.pred_dir..'train_feat.txt',params.train_set.key_list,params.train_set.entity_map)
print(string.format("Train. rep saved in %.2f minutes.",((sys.clock()-start)/60)))
print('Computing dev. rep.')
start=sys.clock()
compute_vectors(params.dev_set,params.pred_dir..'dev_feat.txt')
print(string.format("Dev. rep obtained in %.2f minutes.",((sys.clock()-start)/60)))
print('Computing test. rep.')
start=sys.clock()
compute_vectors(params.test_set,params.pred_dir..'test_feat.txt')
print(string.format("Test. rep obtained in %.2f minutes.",((sys.clock()-start)/60)))
