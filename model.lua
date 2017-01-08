require('nn')
require('cunn')
require('nngraph')

function build_model_multi(i1_row, i1_col, i1_name, i2_row, i2_col, i2_name, dim, csize, tree, root, c_mode)
	local inputs = {nn.Identity()(), nn.Identity()(), nn.Identity()()}
	local i_1 = nn.LookupTable(i1_row, i1_col)(inputs[1]):annotate{name=i1_name}
	local i_2 = nn.LookupTable(i2_row, i2_col)(inputs[2]):annotate{name=i2_name}
	local tar = inputs[3]

	local out = nil
	local out_size = dim
	if c_mode == 1 then
		local c12 = nn.JoinTable(2){i_1, i_2}
		out = nn.View(-1):setNumInputDims(2)(c12)
		out_size = (csize + 1) * dim
	elseif c_mode == 2 then
		out = nn.Sum(2)(nn.JoinTable(2){i_1, i_2})
	elseif c_mode == 3 then
		out = nn.Mean(2)(nn.JoinTable(2){i_1, i_2})
	elseif c_mode == 4 then
		local c1 = nn.View(-1):setNumInputDims(2)(i_1)
		local attention_probs = nn.SoftMax()(nn.Linear(csize * dim, csize)(c1))
		local attention_3d_probs = nn.View(1, -1):setNumInputDims(1)(attention_probs)
		local c11 = nn.MM(false, false){attention_3d_probs, i_1}
		local c12 = nn.CAddTable(){c11, i_2}
		out = nn.View(-1):setNumInputDims(1)(c12)
	end

	local pred = nn.SoftMaxTree(out_size, tree, root){out, tar}

	return nn.gModule(inputs, {pred})
end

-- Function to build frequency-based tree for Hierarchical Softmax
function create_frequency_tree(freq_map)
	binSize=100
	local ft=torch.IntTensor(freq_map)
	local vals,indices=ft:sort()
	local tree={}
	local id=indices:size(1)
	function recursiveTree(indices)
		if indices:size(1)<binSize then
			id=id+1
			tree[id]=indices
			return
		end
		local parents={}
		for start=1,indices:size(1),binSize do
			local stop=math.min(indices:size(1),start+binSize-1)
			local bin=indices:narrow(1,start,stop-start+1)
			assert(bin:size(1)<=binSize)
			id=id+1
			table.insert(parents,id)
			tree[id]=bin
		end
		recursiveTree(indices.new(parents))
	end
	recursiveTree(indices)
	return tree,id
end

function build_model(i1_row, i1_col, i1_name, dim, csize, tree, root, c_mode)
	local inputs = {nn.Identity()(), nn.Identity()()}
	local i_1 = nn.LookupTable(i1_row, i1_col)(inputs[1]):annotate{name=i1_name}
	local tar = inputs[2]

	local out = nil
	local out_size = dim
	if c_mode == 1 then
		out = nn.View(-1):setNumInputDims(2)(i_1)
		out_size = csize * dim
	elseif c_mode == 2 then
		out = nn.Sum(2)(i_1)
	elseif c_mode == 3 then
		out = nn.Mean(2)(i_1)
	elseif c_mode == 4 then
		local c1 = nn.View(-1):setNumInputDims(2)(i_1)
		local attention_probs = nn.SoftMax()(nn.Linear(csize * dim, csize)(c1))
		local attention_3d_probs = nn.View(1, -1):setNumInputDims(1)(attention_probs)
		local c11 = nn.MM(false, false){attention_3d_probs, i_1}
		out = nn.View(-1):setNumInputDims(1)(c11)
	end

	local pred = nn.SoftMaxTree(out_size, tree, root){out, tar}	

	return nn.gModule(inputs, {pred})
end

function build_model_multi_normal(i1_row, i1_col, i1_name, i2_row, i2_col, i2_name, dim, csize, c_mode, output)
	local inputs = {nn.Identity()(), nn.Identity()()}
	local i_1 = nn.LookupTable(i1_row, i1_col)(inputs[1]):annotate{name=i1_name}
	local i_2 = nn.LookupTable(i2_row, i2_col)(inputs[2]):annotate{name=i2_name}

	local out = nil
	local out_size = dim
	if c_mode == 1 then
		local c12 = nn.JoinTable(2){i_1, i_2}
		out = nn.View(-1):setNumInputDims(2)(c12)
		out_size = (csize + 1) * dim
	elseif c_mode == 2 then
		out = nn.Sum(2)(nn.JoinTable(2){i_1, i_2})
	elseif c_mode == 3 then
		out = nn.Mean(2)(nn.JoinTable(2){i_1, i_2})
	elseif c_mode == 4 then
		local c1 = nn.View(-1):setNumInputDims(2)(i_1)
		local attention_probs = nn.SoftMax()(nn.Linear(csize * dim, csize)(c1))
		local attention_3d_probs = nn.View(1, -1):setNumInputDims(1)(attention_probs)
		local c11 = nn.MM(false, false){attention_3d_probs, i_1}
		local c12 = nn.CAddTable(){c11, i_2}
		out = nn.View(-1):setNumInputDims(1)(c12)
	end

	local res = nn.Linear(out_size, output)(out)

	return nn.gModule(inputs, {res})
end

function build_model_normal(i1_row, i1_col, i1_name, dim, csize, c_mode, output)
	local inputs = {nn.Identity()()}
	local i_1 = nn.LookupTable(i1_row, i1_col)(inputs[1]):annotate{name=i1_name}

	local out = nil
	local out_size = dim
	if c_mode == 1 then
		out = nn.View(-1):setNumInputDims(2)(i_1)
		out_size = csize * dim
	elseif c_mode == 2 then
		out = nn.Sum(2)(i_1)
	elseif c_mode == 3 then
		out = nn.Mean(2)(i_1)
	elseif c_mode == 4 then
		local c1 = nn.View(-1):setNumInputDims(2)(i_1)
		local attention_probs = nn.SoftMax()(nn.Linear(csize * dim, csize)(c1))
		local attention_3d_probs = nn.View(1, -1):setNumInputDims(1)(attention_probs)
		local c11 = nn.MM(false, false){attention_3d_probs, i_1}
		out = nn.View(-1):setNumInputDims(1)(c11)
	end

	local res = nn.Linear(out_size, output)(out)

	return nn.gModule(inputs, {res})
end