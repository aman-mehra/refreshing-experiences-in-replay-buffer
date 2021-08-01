import random
import numpy as np

class ReplayBuffer:
    

    def __init__(self, buffer_size=100, batch_sz=10):

        self.buffer_size = buffer_size
        self.batch_sz = batch_sz
        self.pointer = 0
        self.cur_size = 0
        self.buffer = [0 for i in range(buffer_size)]
        self.quorom = min(buffer_size/10,10000)
        

    def at_quorom(self):

        return self.cur_size >= self.quorom
        

    def add(self,transition):

        self.buffer[self.pointer] = transition

        self.pointer = (self.pointer+1)%self.buffer_size

        self.cur_size = min(self.cur_size+1,self.buffer_size)
        

    def sample(self):

        idx = random.randint(0,self.cur_size-1)

        transition = self.buffer[idx]

        return idx, transition
    

    def sample_batch(self):

        b_state,b_action,b_reward,b_next_state,b_done = [], [], [], [], []
        indices = []

        while len(indices) < self.batch_sz:

            idx, transition = self.sample()

            if idx not in indices:

                indices.append(idx)

                state,action,reward,next_state,done = transition

                b_state.append(state)
                b_action.append(action)
                b_reward.append(reward)
                b_next_state.append(next_state)
                b_done.append(done)

        return b_state, b_action, b_reward, b_next_state, b_done

    



class nStepBufferWrapper(ReplayBuffer):

    def __init__(self, gamma=0.99, n=1, buffer_size=100, batch_sz=10, cur_state_history=1):

        ReplayBuffer.__init__(self, buffer_size=buffer_size, batch_sz=batch_sz)

        self.n = n
        self.gamma = gamma
        self.cur_state_history = cur_state_history
        self.n_step_history = [0 for i in range(self.n + 1 + (self.cur_state_history-1))]
        self.n_step_ptr = 0

    # Empty n-step history
    def reset_n_step_buffer(self):

        self.n_step_history = [0 for i in range(self.n + 1 + (self.cur_state_history-1))]

        self.n_step_ptr = 0
        
    """
    Buffer entry comprises - St, A(t-1) and R(t-1)
    """
    def update_n_step_buffer(self,action,next_state,reward):

        self.n_step_history[self.n_step_ptr] = (next_state,action,reward)

        self.n_step_ptr = (self.n_step_ptr+1)%len(self.n_step_history)


    """
    Create state St = {s(t-k+1),s(t-k+2),..s(t-1),s(t)}
    """
    def make_state(self,ptr):

        init_ptr = ptr - (self.cur_state_history-1)
        state = []

        for i in range(self.cur_state_history):

            if isinstance(self.n_step_history[(init_ptr+i)%len(self.n_step_history)], int):
                continue

            s,_,_ = self.n_step_history[(init_ptr+i)%len(self.n_step_history)]

            # Special Case at the start of an episode when num steps < cur_state_history
            while len(state) < i:
                state.append(s)

            state.append(s)

        state = np.stack(state)

        return state

    def make_cur_state(self):

        return self.make_state(self.n_step_ptr-1)


    """
    Computes to n step discounted reward and the state trajectory of the n-step return
    """
    def uncorrected_n_step(self,start_ptr):
        # Reward Accumulator
        reward = 0

        # Computes running discount factor
        cur_gamma = 1

        # Stores trajectory of next states - [ S(t+1), ..., S(t+N)]
        # This is required for the Refresh operation on the intrinsic reward
        next_state_trajectory = []

        for ptr in range(start_ptr+1,start_ptr+1+self.n):

            state,_,r = self.n_step_history[ptr%len(self.n_step_history)]

            # Truncate reward as goal reached
            if state is None: 
                break

            # Updating discounted reward
            reward += r*cur_gamma

            # Updating discount factor
            cur_gamma = cur_gamma*self.gamma

            # Add state to trajectory
            # next_state_trajectory.append(self.make_state(ptr))

        return reward, next_state_trajectory
           

    def make_transition(self):

        start_ptr = (self.n_step_ptr + (self.cur_state_history-1))%len(self.n_step_history)

        # Retrieving start state St
        start_state = self.make_state(start_ptr)

        # Retrieving action At 
        _,action,_ = self.n_step_history[(start_ptr+1)%len(self.n_step_history)]

        # Computing discounted N step reward - Rt + γR(t+1) + ... γ^(N-1)R(t+N-1)
        reward, next_state_trajectory = self.uncorrected_n_step(start_ptr)

        # Retrieving final state S(t+N)
        final_state = self.make_state((start_ptr+self.n)%len(self.n_step_history))

        # Creating transition
        transition = (start_state,action,reward,final_state,next_state_trajectory)

        return transition
        

    def sample_batch(self):

        b_state,b_action,b_reward,b_next_state,b_next_state_trajectory,b_done = [], [], [], [], [], []
        indices = []

        while len(b_state) < self.batch_sz:

            idx, transition = self.sample()

            if idx not in indices:

                indices.append(idx)

                state,action,reward,next_state,next_state_trajectory,done = transition

                b_state.append(state)
                b_action.append(action)
                b_reward.append(reward)
                b_next_state.append(next_state)
                b_done.append(done)
                b_next_state_trajectory.append(next_state_trajectory)
        
        return indices,b_state, b_action, b_reward, b_next_state, b_done, b_next_state_trajectory

    def refresh_transition(self, indices, new_encoded_next, action_new):

        ctr = 0

        for idx in indices:

            state,_,reward,_,next_state_trajectory,done = self.buffer[idx]

            action = action_new[ctr]
            next_state = new_encoded_next[ctr:ctr+1,:]

            self.buffer[idx] = state,action,reward,next_state,next_state_trajectory,done

            ctr += 1
