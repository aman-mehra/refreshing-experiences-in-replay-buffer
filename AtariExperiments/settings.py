config = {


	# Controls Experiment - Uniform for DQN and Refresh for ICM
    "experiment":"Uniform", # "Uniform / Refresh 
    # Buffer capacity control
    "buffer_sz":10000, # / 1000 / 10000
    # N step parameter control
    "n":3, 
    # Learning rate for DQN module
    "lr":0.001, 
    
    # Action Space of Environment
    "action_space":2, 
    # Batch Size for training
    "batch_sz":16, 
    # LEGACY - REMOVE
    "alpha":0.1,
    # Initial Epsilon Value for epsilon greedy policy. Decays linaerly to 0.05
    # TODO - add variable for minimum epsilon value
    "epsilon":1, # default = 0.1
    # Linear reduction in epsilon value at each episode
    "decay_factor":0.0003,
    # Hyperparameter for computing to Return - Rt + gamma * Q(t+1)
    "gamma":0.99, # default = 0.99



    ######### Refresh Settings ################

    # Learning Rates of Different Modules

    # A2C model learning rate
    "a2c_lr":0.001,
    # ICM model leaaring rate
    "icm_lr":0.0003,
    # Feature Base Model learning rate
    "fb_lr":0.001,


    # Update Frequency of Different Modules

    # Training step interval for computing and backpropagating overall actor-critic loss.
    "a2c_update_freq":8,
    # Training step interval for computing and backpropagating critic loss
    # Should be a multiple of a2c_update_freq
    "critic_update_freq":8,
    # Training step interval for computing and backpropagating icm loss
    "icm_update_freq":1,


    # Hyperparameter multipliers for different loss components
    "actor_alpha":1,
    "critic_alpha":0.001,
    "entropy_alpha":0.75,
    "inverse_alpha":1,
    "forward_alpha":0.1,


    # Control Variable to enable/disable buffer refreshing
    "disable_refresh":False,


    # Controls when policy begins training. Trains only feature base and ICM for policy_train_delay episodes
    "policy_train_delay":2000,



    ############ Logging Setting ##############

    # Root Directory for Logging
    "log_path":"./Logs/",

    # Experiment (TODO - Remove Redundancy with 'experiment' parameter above). Also controls Logging sub folder
    "framework":"Refresh", # DQN / Refresh

    # Toggle on or off debugging print statements in code (TODO - output messages can be improved)
    "debug_logging_on":True,


    
    ############# Atari Settings ###############

    # Environment Name
    "game":"CartPole-v0",

    # Determines cropped frame size in image state spaces.
    # For non image based states such as in CartPole, this represents the input state dimension to the neural network
    "crop_sz":8, # 42,

    # LEGACY - REMOVE
    "skip_steps":4,

    # Control type of neural network. Use 'fc' network for non image state spaces, and 'conv' for images
    "net_type":"fc", # conv , fc

    # 'gpu' to enable gpu, 'cpu' otherwise
    "device":"gpu",



    ################ Runtime Constants ############

    # Total number of episodes to run for
    "EPISODES":8000,

    # End an epsiode if after TIMEOUT steps it hasn't terminated
    "TIMEOUT" :5000,

    # Decay learning rate every 'lr_decay_freq' episodes
    "lr_decay_freq":3000, 

    # For DQN only. Synchronize target network weights every 'sync_network_steps' training steps
    "sync_network_steps":5,

    # Skip 'skip_frames' consecutive frames from environment for every frame stores into replay buffer
    "skip_frames":1,

    # Stick to the chosen action for 'sticky_action_freq' environment steps
    "sticky_action_freq":1,

    # Current state consists of the currently observed state +  the last 'cur_state_history - 1' observed states
    "cur_state_history":2,

    # Train network ever 'train_step_freq' environment steps
    "train_step_freq": 64,

    # When training network in a particular environment step, train 'train_iter_together' batches sequentially
    "train_iter_together":4,

}


# config = {
#     "experiment":"Uniform", # "Uniform / Refresh / Refresh-Noisy / Geometric-Decay / Small-Big / CER / CER-Refresh / PER
#     "buffer_sz":100000, # / 1000 / 10000
#     "n":1, # 1 / 2 / 3
#     "lr":0.001, # 
    
#     "action_space":4, # default = 4
#     "batch_sz":32, # default = 10
#     "alpha":0.1, # default = 0.1
#     "epsilon":1, # default = 0.1
#     "gamma":0.99, # default = 0.99

#     # Logging Setting
#     "log_path":"./Logs/",
#     "framework":"DQN", # DQN / Refresh
    
#     # Atari Settings
#     "game":"Pong-v0",
#     "crop_sz":84, # 42,
#     "skip_steps":4,
#     "net_type":"conv", # conv , fc

#     # Training settings
#     "device":"gpu",

#     # Runtime CONSTANTS
#     "EPISODES":15000,
#     "DECAY_FREQ":50,
#     "TIMEOUT" :20000,
#     "sync_network_steps":1000,
#     "skip_frames":4,
#     "sticky_action_freq":1,
#     "cur_state_history":3,
#     "train_step_freq":64,
#     "train_iter_together":4,
#     "decay_factor":0.0003,

# }

