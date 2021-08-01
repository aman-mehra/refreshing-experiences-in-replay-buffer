import gym
import cv2
import random

class GymWrapper:

	def __init__(self, game, crop_sz = 84):
		self.game = game
		self.env = gym.make(game)
		self.crop_sz = crop_sz

	def process_state(self,state):
		# Process Images only
		if len(state.shape) > 1:
			state = cv2.cvtColor(cv2.resize(state, (self.crop_sz,self.crop_sz)), cv2.COLOR_RGB2GRAY)/255.
			# idx = random.randint(0,int(21*self.crop_sz//16.0)-self.crop_sz)
			# state = state[idx:idx+self.crop_sz,:]
		return state

	def sample_action(self):
		return self.env.action_space.sample()

	def step(self, action):
		state, reward, done, info = self.env.step(action)
		state = self.process_state(state)
		return state, reward, done, info

	def reset(self):
		return self.process_state(self.env.reset())

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()

