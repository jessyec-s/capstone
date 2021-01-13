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
    def __init__(self,beta,input_dims,n_actions,name='critic',fc1_dims=256,fc2_dims=256,chkpt_dir='tmp/sac'):
        super(CriticNetwork,self).__init__()
        # save parameters
        self.input_dims=input_dims
        self.n_actions=n_actions
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        # we have both
        self.checkpoint_dir=chkpt_dir
        self.checkpoint_file=os.path.join(self.checkpoint_dir,name+'_sac')
        self.name=name

        # define neural network
        # linear transformation y=xA_T +b or input_dims+n_actions*fc1_dims
        # add actions to input dims, then matrix multiply with output dims
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        # matrix multiply with second layer
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        # matrix multiply with third layer (this isn't happening yet, we are just setting up the network)
        self.q = nn.Linear(self.fc2_dims,1)
        # set up module; parameters are for optimizations
        self.optimizer = optim.Adam(self.parameters(),lr=beta)

        # this is about setting device to
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state,action):
        # propagate action value through the neural network
        action_value=self.fc1(T.cat([state,action],dim=1))
        action_value=F.relu(action_value)
        action_value=self.fc2(action_value)
        q = self.q(action_value)
        return q
    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self,beta,input_dims,name='value',fc1_dims=256,fc2_dims=256,chkpt_dir='tmp/sac'):
        super(ValueNetwork,self).__init__()
        # save parameters
        self.input_dims=input_dims
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        # we have both
        self.checkpoint_dir=chkpt_dir
        self.checkpoint_file=os.path.join(self.checkpoint_dir,name+'_sac')
        self.name=name

        # define neural network
        # linear transformation y=xA_T +b or input_dims+n_actions*fc1_dims
        # add actions to input dims, then matrix multiply with output dims
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # matrix multiply with second layer
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        # matrix multiply with third layer (this isn't happening yet, we are just setting up the network)
        self.v = nn.Linear(self.fc2_dims,1)
        # set up module; parameters are for optimizations
        self.optimizer = optim.Adam(self.parameters(),lr=beta)

        # this is about setting device to
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        # pass through
        state_value=self.fc1(state)
        state_value=F.relu(state_value)
        state_value=self.fc2(state_value)
        state_value=F.relu(state_value)
        v=self.v(state_value)
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# handle sample distribution rather than feed forward
class ActorNetwork(nn.Module):
    def __init__(self,alpha, input_dims,max_action, fc1_dims=256, fc2_dims=256,n_actions=2, name='actor',chkpt_dir='tmp/sac'):
        super(ActorNetwork,self).__init__()
        self.input_dims=input_dims
        self.max_action=max_action
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.n_actions=n_actions
        self.name=name

        self.checkpoint_dir=chkpt_dir
        self.checkpoint_file=os.path.join(self.checkpoint_dir,name+'_sac')
        self.max_action = max_action

        self.reparam_noise=1e-6
        #now define deep nn
        self.fc1=nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2=nn.Linear(self.fc1_dims,self.fc2_dims)
        # mean and stdev
        self.mu=nn.Linear(self.fc2_dims,self.n_actions)
        self.sigma=nn.Linear(self.fc2_dims,self.n_actions)

        self.optimizer=optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        prob=self.fc1(state)
        prob=F.relu(prob)
        prob=self.fc2(prob)
        prob=F.relu(prob)
        mu=self.mu(prob)
        sigma = self.sigma(prob)
        # need to clamp sigma
        sigma = T.clamp(sigma,min=self.reparam_noise,max=1)
        return mu,sigma

    def sample_normal(self,state, reparameterize=True):
        mu, sigma =self.forward(state)
        probabilities= Normal(mu,sigma)
        if reparameterize:
            # adding some noise to actions
            actions=probabilities.rsample()
        else:
            actions=probabilities.sample()
        action=T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        # log loss function
        log_probs = probabilities.log_prob(actions)
        log_probs-=T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1,keepdim=True)

        return action,log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
