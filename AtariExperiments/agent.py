import numpy as np
import cv2
import random
from models import *
import utils 
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from settings import *


class RefreshAgent:
    
    def __init__(self,actor_alpha=0.5,entropy_alpha=0.2,inverse_alpha=0.07,forward_alpha=1.5,epsilon=0.1,base_epsilon=0.8,min_epsilon=0.1,device="cuda"):
        self.device = torch.device(device)
        
        if config["net_type"] == "fc":
            self.icm = ICMModule(config["action_space"],config["net_type"]).to(device)
            self.a2c = A2C(config["action_space"],config["net_type"]).to(device)
            self.feature_base = FeatureExtractor(config["crop_sz"],config["net_type"]).to(device)
        elif config["net_type"] == "conv":
            self.icm = ICMModule(config["action_space"],config["net_type"]).to(device)
            self.a2c = A2C(config["action_space"],config["net_type"]).to(device)
            self.feature_base = FeatureExtractor(config["cur_state_history"],config["net_type"]).to(device)

        self.pred_action_criterion = nn.CrossEntropyLoss()
        self.pred_state_criterion = nn.MSELoss()

        self.critic_loss = nn.MSELoss()
        
        self.optimizer_icm = torch.optim.Adam(self.icm.parameters(), lr=config["icm_lr"])
        self.optimizer_a2c = torch.optim.Adam(self.a2c.parameters(), lr=config["a2c_lr"])
        self.optimizer_fb = torch.optim.Adam(self.feature_base.parameters(), lr=config["fb_lr"])

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.base_epsilon = base_epsilon

        # Controls Training of Various Submodules
        self.train_policy = False
        self.begin_refresh = False
        self.sync_ctr = 1


    def update_agent(self,episode):
        self._update_training_strategy(episode)
        self._update_learning_rate(episode)
        self._update_epsilon()


    def _update_training_strategy(self,episode):

        if episode >= config["policy_train_delay"]:
            self.train_policy = True
            self.begin_refresh = True

            for param in self.optimizer_icm.param_groups:
                param['lr'] = param['lr']/20
            for param in self.optimizer_a2c.param_groups:
                param['lr'] = param['lr']/20


    def _update_learning_rate(self,episode):
        for param in self.optimizer_icm.param_groups:
            param['lr'] = max(param['lr']*(0.5**(episode//config["lr_decay_freq"])),0.00001)
        for param in self.optimizer_a2c.param_groups:
            param['lr'] = max(param['lr']*(0.5**(episode//config["lr_decay_freq"])),0.00001)
        for param in self.optimizer_fb.param_groups:
            param['lr'] = max(param['lr']*(0.5**(episode//config["lr_decay_freq"])),0.00001)
      

    def _update_epsilon(self):
        self.epsilon = max(self.epsilon - config["decay_factor"],self.min_epsilon)


    def epsilon_greedy_action(self,state):

        if random.random() < self.epsilon:
            return random.randint(0,config["action_space"]-1)

        with torch.no_grad():

            if config["net_type"] == "fc":
                state = state.reshape((1,-1))

            input = torch.from_numpy(state).type(torch.FloatTensor).to(self.device).unsqueeze(0)
            features = self.feature_base(input)
            action_probs, _, _ = self.a2c(features,None,predict=True)

            greedy_action = np.argmax(action_probs.detach().cpu().numpy())

            return greedy_action


    def learn(self,batch):

        state,action,reward,encoded_next_state = utils.make_batch_tensors(batch,"refresh",config["net_type"])

        state = state.to(self.device)
        encoded_next_state = encoded_next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)

        self.optimizer_icm.zero_grad()
        self.optimizer_a2c.zero_grad()
        self.optimizer_fb.zero_grad()

        encoded_state = self.feature_base(state)

        # ICM Forward

        action_one_hot = torch.zeros(action.shape[0], config["action_space"])
        action_one_hot[torch.arange(action.shape[0]), action] = 1
        action_one_hot = action_one_hot.to(self.device)

        pred_encoded_next, pred_action = self.icm((encoded_state,encoded_next_state,action_one_hot))

        loss_inverse = config["inverse_alpha"]*self.pred_action_criterion(pred_action,action.squeeze(1))
        loss_forward = config["forward_alpha"]*self.pred_state_criterion(pred_encoded_next,encoded_next_state)


        # A2C Forward

        trgt_variance = 0

        if self.train_policy:

            action_outputs, cur_state_value, next_state_value = self.a2c(encoded_state,encoded_next_state)

            action_probs = F.softmax(action_outputs,dim=1)
            log_probs = F.log_softmax(action_outputs,dim=1)
            action_dist = Categorical(action_probs)    
            action_new = action_dist.sample()
            log_action_probs = log_probs.gather(1,action_new.unsqueeze(-1))

            intrinsic_reward = self.critic_loss(pred_encoded_next,encoded_next_state).detach()
            target = intrinsic_reward + reward + (config["gamma"]**config["n"])*next_state_value


            actor_loss = (log_action_probs * (target-cur_state_value)).mean()

            critic_loss = 0
            if self.sync_ctr%config["critic_update_freq"]==0:
                critic_loss = self.critic_loss(cur_state_value,target)

            entropy_loss = (- log_probs * action_probs).mean()

            a2c_loss = config["critic_alpha"]*critic_loss - config["actor_alpha"]*actor_loss - config["entropy_alpha"]*entropy_loss

            trgt_variance = torch.var(target).item()#detach().cpu().numpy()[0]
            
            if config["debug_logging_on"]:
                if random.random()<0.05:
                    if not isinstance(critic_loss,int):
                        print(config["critic_alpha"]*critic_loss.detach().cpu().numpy(),config["actor_alpha"]*actor_loss.detach().cpu().numpy(),config["entropy_alpha"]*entropy_loss.detach().cpu().numpy())
                    print(config["inverse_alpha"]*loss_inverse.detach().cpu().numpy(),config["forward_alpha"]*loss_forward.detach().cpu().numpy(),a2c_loss.detach().cpu().numpy())

            if self.sync_ctr % config["a2c_update_freq"] == 0:

                a2c_loss.backward(retain_graph=True)

                for param in self.a2c.parameters():
                    param.grad.data.clamp_(-1,1)

                self.optimizer_a2c.step()


        if self.sync_ctr % config["icm_update_freq"] == 0:

            loss_inverse.backward(retain_graph=True)
            loss_forward.backward(retain_graph=True)

            for param in self.icm.parameters():
                param.grad.data.clamp_(-1,1)

            self.optimizer_icm.step()

        for param in self.feature_base.parameters():
            param.grad.data.clamp_(-1,1)

        self.optimizer_fb.step()

        self.sync_ctr += 1


        # Refresh

        if self.begin_refresh:

            new_action_one_hot = torch.zeros(action.shape[0], config["action_space"])
            new_action_one_hot[torch.arange(action.shape[0]), action_new] = 1
            new_action_one_hot = new_action_one_hot.to(self.device)

            new_encoded_next, _ = self.icm((encoded_state,encoded_next_state,new_action_one_hot),predict=False)

            new_encoded_next = new_encoded_next.detach().cpu().numpy()
            action_new = action_new.detach().cpu().numpy()

            return new_encoded_next, action_new, trgt_variance

        else:

            new_encoded_next = encoded_next_state.detach().cpu().numpy()
            action_new = action.detach().cpu().numpy().reshape((action.shape[0]))
            
            return new_encoded_next, action_new, trgt_variance 


class DQNAgent:
    
    def __init__(self,input_dim=84,net_type="conv",lr=0.001,epsilon=0.1,base_epsilon=1,min_epsilon=0.05,device="cuda"):
        self.device = torch.device(device)

        self.net_type = net_type
        if net_type == "fc":
            self.net = DQN_FC(config["action_space"],input_dim).to(device)
            self.trgt = DQN_FC(config["action_space"],input_dim).to(device)
        elif net_type == "conv":
            self.net = DQN(config["action_space"],input_dim,in_channels=config["cur_state_history"]).to(device)
            self.trgt = DQN(config["action_space"],input_dim,in_channels=config["cur_state_history"]).to(device)

        self.trgt.eval()
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=0.00025, alpha=0.99, eps=1e-06)

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.base_epsilon = base_epsilon

        self.sync_ctr = 1


    def update_learning_rate(self,episode):
        for param in self.optimizer.param_groups:
            param['lr'] = max(param['lr']*(0.5**(episode//config["lr_decay_freq"])),0.00001)

        
    """
    Periodically update epsilon. This function linearly decays epsilon from self.base_epsilon to self.min_epsilon
    using a step size of config["decay_factor"]. Epsilon is udated every episode.
    """
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - config["decay_factor"],self.min_epsilon)


    """
    Pick an action using an epsilon greedy policy
    """
    def epsilon_greedy_action(self,state):

        # Sample Random Action with probability = epsilon
        if random.random() < self.epsilon:
            return random.randint(0,config["action_space"]-1)

        # Sample greedy action from policy with probability = 1- epsilon
        with torch.no_grad():

            # Form appropriate input dimension for FC Network 
            if self.net_type == "fc":
                state = state.reshape((1,-1))

            # Obtain Q values for given input state
            input = torch.from_numpy(state).type(torch.FloatTensor).to(self.device).unsqueeze(0)

            q_values = self.net(input)

            # Find greedy action
            greedy_action = np.argmax(q_values.detach().cpu().numpy())

            return greedy_action


    """
    Trains a batch of experiences on the DQN Network
    """
    def learn(self,batch):
        # Creating training batch tensors
        state,action,reward,next_state = utils.make_batch_tensors(batch,mode="regular",net_type=self.net_type)

        # Moving tensors to exection device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)

        self.optimizer.zero_grad()

        # DQN Forward Pass

        # Computing Q(st,at)
        state_action_values = self.net(state).gather(1,action).squeeze(-1)

        # Computing Q(st+1,at+1)
        # This is computed using the target network
        next_state_action_values = self.trgt(next_state).max(1)[0]
        next_state_action_values = next_state_action_values.detach()

        # Computing target = Gt + (config["gamma"]^n)*Q(st+1,at+1)
        target = reward[:,0] + (config["gamma"]**(config["n"]))*next_state_action_values

        # Computing DQN Loss and Backpropagating
        dqn_loss = F.smooth_l1_loss(state_action_values,target)
        dqn_loss.backward()

        for param in self.net.parameters():
            param.grad.data.clamp_(-1,1)

        self.optimizer.step()

        trgt_variance = torch.var(target).item()
        # trgt_variance = dqn_loss.item()

        # Syncing target network weights every config["sync_network_steps"] steps
        if self.sync_ctr % config["sync_network_steps"] == 0:
            self.trgt.load_state_dict(self.net.state_dict())

        self.sync_ctr += 1

        return None,None,trgt_variance











