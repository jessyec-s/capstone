import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from ddpgnetwork import ActorNetwork, CriticNetwork
from OUP import OrnsteinUhlenbeckProcess
from normalizer import Normalizer

class Agent():
    def __init__(self,n_states, n_actions, n_goals, alpha=0.0001,beta=.001,input_dims=[8], gamma=.99,n_actions=2,
                 max_size=1000000,layer1_size=256,layer2_size=256,tau=.005,batch_size=256,reward_scale=2, k_future=4):
        # reward scales  depends on action convention for the environment\

        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.tau=tau
        self.k_future=k_future
        self.n_states=n_states
        self.n_actions=n_actions
        self.n_goals=n_goals

        # Create the networks
        self.actor=ActorNetwork(alpha, input_dims,n_actions=n_actions)
        self.critic=CriticNetwork(beta,input_dims,n_actions=n_actions)
        self.target_actor=ActorNetwork(alpha,input_dims,n_actions=n_actions)
        self.target_critic=CriticNetwork(beta,input_dims,n_actions=n_actions)

        # Create the optimizers
        self.actor_optimizer = T.optim.Adam(self.actor.parameters(),lr=self.alpha)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(),lr=self.beta)

        # Initialize the target networks
        self.hard_update(self.target_actor,self.actor)
        self.hard_update(self.target_critic,self.critic)

        # Initialize buffer and noise 
        self.memory = ReplayBuffer(max_size,input_dims,n_actions,k_future)
        self.random = OrnsteinUhlenbeckProcess(size=n_actions, theta=.15, mu=0.0,sigma=.2)

        # Initialize normalizers
        self.state_normalizer = Normalizer(self.n_states[0], default_clip_range=5)
        self.goal_normalizer = Normalizer(self.n_goals, default_clip_range=5)

        self.batch_size=batch_size
        self.s_t=None
        self.a_t=None

    def hard_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def choose_action(self,observations,desired_goals):
        # here we turn into a tensor
        # we might need to normalize here
        observations = self.state_normalizer.normalize(observation)
        goals = self.goal_normalizer.normalize(desired_goal)
        observations = T.tensor(observation, dtype=T.float).to(self.actor.device)
        goals = T.tensor(goal, dtype=T.float).to(self.actor.device)
        input_tensor = torch.cat([observations, goals], dim=1)

        action = self.actor(input_tensor)
        # print(action," d "  ,action.detach().numpy(),"  d ",self.random.sample())
        return action #action.detach().numpy()+self.random.sample()

    def remember(self,state,action,reward,new_state,done):
        self.memory.store_transition(state,action,reward,new_state,done)

    def update_network_parameters(self,tau=None):
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)

   #depending on models, we need to save or load
    def save_models(self):
        print("saving models:")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.target_actor.save_checkpoint()

    def load_models(self):
        print("loading models:")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.target_actor.load_checkpoint()

    def  learn(self):
        #  must fully load up memory, otherwise must keep learning
        # if self.memory.mem_cntr <self.batch_size:
        #     return

        state,action, reward,new_state, done = self.memory.sample_buffer(self.batch_size)

        reward_batch = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done_batch = T.tensor(done).to(self.actor.device)
        next_state_batch = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state_batch = T.tensor(state, dtype=T.float).to(self.actor.device)
        action_batch = T.tensor(action,dtype=T.float).to(self.actor.device)

        next_q_values = self.target_critic(next_state_batch,self.target_actor(next_state_batch))
        # print("next q batch",next_q_values.size())
        # print("reward batch", reward_batch.size(), " done_batch",done_batch.unsqueeze(1).size(),next_q_values.size())

        target_q_batch= reward_batch.unsqueeze(1) +self.gamma*(~done_batch).unsqueeze(1)*next_q_values
        # print("target q batch",target_q_batch.size())
        #critic update

        self.critic_optimizer.zero_grad()
        q_batch = self.critic(state_batch,action_batch)

        value_loss = F.mse_loss(q_batch,target_q_batch)
        value_loss.backward()

        self.critic_optimizer.step()
        # actor update
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch,self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
