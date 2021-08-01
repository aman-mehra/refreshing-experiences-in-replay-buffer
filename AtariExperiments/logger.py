import cv2
import matplotlib.pyplot as plt
import pickle
import os

class Logger:
    
    def __init__(self,base_path,path,name,episodes):
        self.path = base_path + path
        self.name = name
        self.episodes = episodes
        self.ep_stats = {}
        self.logs={}
        self.create()
        self.setup()
        
    def create(self):
        os.makedirs(self.path,exist_ok=True)
        open(self.path+self.name, 'ab').close()
        
                
    def setup(self):
        self.logs["returns"] = []
        self.logs["running_variance"] = []
        
    def step(self):
        self.logs["returns"].append(self.ep_stats["return"])
        self.logs["running_variance"].append(self.ep_stats["running_variance"])
        
    def save(self):
        with open(self.path+self.name,"wb") as f:
            pickle.dump(self.logs,f)
            
    def load(self):
        self.logs = pickle.load(f)

    def reset_episode_log(self):
        self.ep_stats["return"] = 0
        self.ep_stats["running_variance"] = 0
        self.ep_stats["train_step"] = 0
        
    def update_episode_log(self,reward,running_variance):
        self.ep_stats["train_step"] += 1
        self.ep_stats["return"] += reward
        if running_variance is not None:
            self.ep_stats["running_variance"] += running_variance
        
    def process_episode_log(self):
        self.ep_stats["running_variance"] = self.ep_stats["running_variance"]/self.ep_stats["train_step"]     
