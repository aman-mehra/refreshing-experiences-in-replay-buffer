import numpy as np
import torch

def make_batch_tensors(batch, mode="refresh", net_type="conv"):

	# handling n step barches with trajectory
	if mode == "refresh": 
		_,state,action,reward,encoded_next_state,_,_ = batch

		batch_sz = len(state)

		action = np.stack(action)
		reward = np.stack(reward)
		encoded_next_state = np.stack(encoded_next_state)

		action = torch.from_numpy(action).type(torch.LongTensor).unsqueeze(1)
		reward = torch.from_numpy(reward).type(torch.FloatTensor).unsqueeze(1)
		encoded_next_state = torch.from_numpy(encoded_next_state).squeeze(1)

		if net_type == "conv":

			state = np.stack(state)

		elif net_type == "fc":

			state = np.stack(state).reshape((batch_sz,-1))
			encoded_next_state = encoded_next_state.squeeze(1)

		state = torch.from_numpy(state).type(torch.FloatTensor)

		return state,action,reward,encoded_next_state

	else: 

		_,state,action,reward,next_state,_,_ = batch

		batch_sz = len(state)

		action = np.stack(action)
		reward = np.stack(reward)

		action = torch.from_numpy(action).type(torch.LongTensor).unsqueeze(1)
		reward = torch.from_numpy(reward).type(torch.FloatTensor).unsqueeze(1)

		if net_type == "conv":

			state = np.stack(state)
			next_state = np.stack(next_state)

		elif net_type == "fc":

			state = np.stack(state).reshape((batch_sz,-1))
			next_state = np.stack(next_state).reshape((batch_sz,-1))

		state = torch.from_numpy(state).type(torch.FloatTensor)
		next_state = torch.from_numpy(next_state).type(torch.FloatTensor)

		return state,action,reward, next_state


def get_logging_path(config,method):
	path = config["framework"]+"/"+config["game"]+"/"+config['experiment']+"/"+method+"/"+str(config['n'])+"/"
	return path