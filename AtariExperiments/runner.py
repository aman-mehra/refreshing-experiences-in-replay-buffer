import gym
import torch
from tqdm import tqdm

from buffer import *
from agent import *
from env_wrapper import *
from logger import *
from settings import *
from utils import *


class Runner:
        
    def setup(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "Refresh" in config["experiment"]:
            self.agent = RefreshAgent(epsilon=config['epsilon'], device=self.device)
        else:
            self.agent = DQNAgent(config["crop_sz"], net_type=config["net_type"], lr=config["lr"], epsilon=config['epsilon'], device=self.device)
        
        self.memory = ReplayBuffer(buffer_size=config['buffer_sz'],batch_sz=config['batch_sz'])
        self.env = GymWrapper(config["game"],crop_sz=config["crop_sz"])
        self.logger = Logger(config["log_path"], get_logging_path(config,self.method), str(config['buffer_sz'])+".pickle",config["EPISODES"])
                    

    def save_q_values(self):
        with open(config["log_path"]+get_logging_path(config,self.method)+str(config['buffer_sz'])+"-q.pickle","wb") as f:
            pickle.dump(self.agent.net,f)


    def make_transition(self):

        start_state,action,reward,final_state,next_state_trajectory = self.memory.make_transition()

        if final_state is None:
            final_state = start_state

        encoded_next_state = final_state

        if 'Refresh' in config['experiment']:
            input = torch.from_numpy(final_state.reshape((-1))).type(torch.FloatTensor).to(self.device).unsqueeze(0)
            encoded_next_state = self.agent.feature_base(input).detach().cpu().numpy()

        # Creating transition
        transition = (start_state,action,reward,encoded_next_state,next_state_trajectory)

        return transition
            

class nStepRunner(Runner):
    
    def __init__(self, config):
        self.method = "TD"
        Runner.__init__(self)
    
    def setup(self):
        Runner.setup(self)
        del self.memory
        self.memory = nStepBufferWrapper(config["gamma"],config["n"],config['buffer_sz'],config['batch_sz'],cur_state_history=config["cur_state_history"])
        
    def get_TD_n_target(self,reward,q_next):
        target = reward + (config["gamma"]**config["n"])*q_next
        return target        
        
    
    def train_on_batch(self,batch):

        # Initialize batch statistics
        running_variance = 0

        # Retrive buffer transition indices if refresh required
        indices,_,_,_,_,_,_ = batch

        # Train on batch
        new_encoded_next, action_new, running_variance = self.agent.learn(batch)

        # Refresh Experience Replay
        if not config['disable_refresh']:
            self.memory.refresh_transition(indices, new_encoded_next, action_new)
            
        return running_variance


    """
    Training loop for agent utilizing n-step returns
    """
    def learn_n_step(self):
                
        total_steps = 0
            
        for eps in range(config["EPISODES"]):

            cur_state = self.env.reset()

            self.logger.reset_episode_log()
            self.memory.reset_n_step_buffer()
            self.memory.update_n_step_buffer(0,cur_state,-1)
            self.agent.update_agent(eps)

            T = config["TIMEOUT"]
            done = False
            reward = 0
            time_step = 0
            
            pbar = tqdm(total=config["TIMEOUT"])
            pbar.set_description("Episode "+str(eps+1))

            while True:

                # Not timeout or termination
                if time_step < T:

                    # Pick new action every sticky_action_freq env steps
                    if time_step % config["sticky_action_freq"] == 0:

                        # If memory less than quorom then cannot start training so take random action
                        # Applicable only in the first few episodes
                        if not self.memory.at_quorom():
                            action = random.randint(0,config['action_space']-1)
                        else:
                            action = self.agent.epsilon_greedy_action(self.memory.make_cur_state())

                    # Take action in environment and update n-step history buffer
                    next_state, reward, done, info = self.env.step(action)

                    # Skip frames from consideration
                    for frame_ct in range(config["skip_frames"]-1):
                        next_state, r, done, info = self.env.step(action)
                        reward += r

                    # if total_steps% 10 == 0:
                    # cv2.imwrite("./log_output/"+str(total_steps)+".png",cv2.resize(cur_state*255,(300,300)))

                    cur_state = next_state
                    self.memory.update_n_step_buffer(action,next_state,reward)

                    # reached terminal state
                    if done:
                        T = time_step

                    # incrementing total env steps
                    total_steps += 1

                    # Increment epside env steps
                    time_step += 1
                    pbar.update(1)
                
                # Not timeout - CHECK WHY THIS IS HERE
                # elif time_step != config["TIMEOUT"]:
                #     time_step += 1    
                #     self.memory.update_n_step_buffer(-1,None,0)
                        
                tau = time_step - (config['n'] - 1) - (config["cur_state_history"] )
                
                # Start populating replay memory after a delay of N iterations each episode
                if tau >= 0:

                    # Create transition from past n experiences
                    transition = (*self.make_transition(),done)

                    # Add transition to replay memory
                    self.memory.add(transition)
                    
                    # Skip training if replay memory not sufficiently populated
                    if not self.memory.at_quorom():
                        continue

                    # Train every train_step_freq steps
                    if tau % config["train_step_freq"] == 0:

                        running_variance = 0

                        for train_iter in range(config["train_iter_together"]):

                            # Sample batch of transitions
                            batch = self.memory.sample_batch()

                            # If training with CER, add last added transition to batch
                            # if 'CER' in config["experiment"]:
                            #     batch[-1] = transition

                            # Train on sampled batch
                            running_variance += self.train_on_batch(batch)

                        # Update log with running statistics
                        self.logger.update_episode_log(reward,running_variance/float(config["train_iter_together"]))

                    else:

                        # Update log with running statistics
                        self.logger.update_episode_log(reward,None)
                  
                # Exit loop on episode end  
                if time_step >= T:
                    break

            pbar.set_description("Episode "+str(eps+1)+" Reward "+str(self.logger.ep_stats["return"])+ " Steps "+str(total_steps))
            pbar.close()

            # For the edge case that the minimum memory requirement is not met even after one episode
            if not self.memory.at_quorom():
                continue

                    
            self.logger.process_episode_log()
            self.logger.step()
            self.logger.save()

        self.save_q_values()