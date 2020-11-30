# ddpg doesn't work as well as the other two

import os
import torch as T
import torch.nn.functional as F
import  torch.nn as nn
import torch.optim as optim
from torch.distributions.normal  import Normal

import numpy as np
# standard inheritance

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return T.Tensor(size).uniform_(-v, v)

class CriticNetwork(nn.Module):
    def __init__(self,critic_learning_rate,input_dims,n_actions,name='critic',fc1_dims=256,fc2_dims=256,fc3_dims=256,chkpt_dir='tmp/ddpg',init_w=3e-3):
        super(CriticNetwork,self).__init__()
        # save parameters
        self.name=name
        self.critic_learning_rate=critic_learning_rate
        self.checkpoint_dir=chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')

        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.fc1 = nn.Linear(input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims+n_actions,fc2_dims)
        self.fc3 = nn.Linear(fc2_dims,fc3_dims)
        self.output = nn.Linear(fc3_dims,1)

        self.bn1 = nn.LayerNorm(fc1_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.bn3 = nn.LayerNorm(fc3_dims)

        self.init_weights(init_w)

        # this is about setting device to
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self,state,action):
        # propagate action value through the neural network
        out = self.fc1(state)
        out = F.relu(out)
        out = self.bn1(out)
        out = self.fc2(T.cat([out,action],1))
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        q_val = self.output(out)
        return q_val

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self,actor_learning_rate, input_dims,fc1_dims=256, fc2_dims=256,fc3_dims=256,n_actions=2, name='actor',chkpt_dir='tmp/ddpg',init_w=3e-3):
        super(ActorNetwork,self).__init__()
        self.actor_learning_rate=actor_learning_rate
        self.name=name
        self.checkpoint_dir=chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')


        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.fc3_dims=fc3_dims
        self.fc1= nn.Linear(input_dims,fc1_dims)
        self.fc2= nn.Linear(fc1_dims,fc2_dims)
        self.fc3= nn.Linear(fc2_dims,fc3_dims)
        self.output = nn.Linear(fc3_dims, n_actions)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.tanh = nn.Tanh()
        self.init_weights(init_w)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        out = self.fc1(state)
        out = F.relu(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.bn2(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.bn3(out)
        out = self.output(out)
        out = self.tanh(out)
        return out

    def sample_normal(self,state, reparameterize=True):
        pass

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
