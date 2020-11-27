# ddpg doesn't work as well as the other two

import os
import torch as T
import torch.nn.functional as F
import  torch.nn as nn
import torch.optim as optim
from torch.distributions.normal  import Normal

import numpy as np
# standard inheritance
class CriticNetwork(nn.Module):
    def __init__(self,critic_learning_rate,input_dims,n_actions,name='critic',fc1_dims=256,fc2_dims=256,chkpt_dir='tmp/ddpg'):
        super(CriticNetwork,self).__init__()
        # save parameters


        # this is about setting device to
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state,action):
        # propagate action value through the neural network
        pass

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self,actor_learning_rate, input_dims,max_action, fc1_dims=256, fc2_dims=256,n_actions=2, name='actor',chkpt_dir='tmp/ddpg'):
        super(ActorNetwork,self).__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        pass

    def sample_normal(self,state, reparameterize=True):
        pass

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
