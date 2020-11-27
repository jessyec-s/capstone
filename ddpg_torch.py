import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from network import ActorNetwork, CriticNetwork,ValueNetwork

class Agent():
    def __init__(self,alpha=0.0003,beta=.0003,input_dims=[8],env=None, gamma=.99,n_actions=2,
                 max_size=1000000,layer1_size=256,layer2_size=256,tau=.005,batch_size=256,reward_scale=2):
        # reward scales  depends on action convention for the environment
        pass
    def choose_action(self,observation):
        # here we turn into a tensor
        pass

    def remember(self,state,action,reward,new_state,done):
        self.memory.store_transition(state,action,reward,new_state,done)

    def update_network_parameters(self,tau=None):
        pass
   #depending on models, we need to save or load
    def save_models(self):
        print("saving models:")
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
    def load_models(self):
        print("loading models:")
        self.actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()

    def  learn(self):
        #  must fully load up memory, otherwise must keep learning
        if self.memory.mem_cntr <self.batch_size:
            return
        state,action, reward,new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action,dtype=T.float).to(self.actor.device)


