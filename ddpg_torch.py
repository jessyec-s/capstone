import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from ddpgnetwork import ActorNetwork, CriticNetwork
from OUP import OrnsteinUhlenbeckProcess
class Agent():
    def __init__(self,alpha=0.0001,beta=.001,input_dims=[8], gamma=.99,n_actions=2,
                 max_size=1000000,layer1_size=256,layer2_size=256,tau=.005,batch_size=256,reward_scale=2,env=None,load_checkpoint=False,epochs=40):
        # reward scales  depends on action convention for the environment\

        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.tau=tau

        self.env=env
        self.load_checkpoint=load_checkpoint
        self.epochs=epochs
        self.episodes=50
        self.actor=ActorNetwork(alpha, input_dims,n_actions=n_actions)
        self.critic=CriticNetwork(beta,input_dims,n_actions=n_actions)
        self.target_actor=ActorNetwork(alpha,input_dims,n_actions=n_actions)
        self.target_critic=CriticNetwork(beta,input_dims,n_actions=n_actions)

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(),lr=self.alpha)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(),lr=self.beta)

        self.hard_update(self.target_actor,self.actor)
        self.hard_update(self.target_critic,self.critic)

        desired_size=env.observation_space["desired_goal"].shape[0]
        achieved_size=env.observation_space["achieved_goal"].shape[0]
        
        self.memory = ReplayBuffer(max_size,self.episodes,input_dims,n_actions,desired_size,achieved_size)
        self.random = OrnsteinUhlenbeckProcess(size=n_actions, theta=.15, mu=0.0,sigma=.2)


        self.batch_size=batch_size
        self.s_t=None
        self.a_t=None

    def hard_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def choose_action(self,observation):
        # here we turn into a tensor
        observation= T.tensor(observation, dtype=T.float).to(self.actor.device)
        action = self.actor(observation)
        # print(action," d "  ,action.detach().numpy(),"  d ",self.random.sample())
        return action.detach().numpy()+self.random.sample()

    def remember(self,state,action,reward,new_state,done):
        self.memory.store_transition(state,action,reward,new_state,done)

    def update_network_parameters(self,tau=None):
        pass
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

    def train(self):
        episodes=2
        best_score = self.env.reward_range[0]
        score_history = []

        if self.load_checkpoint:
            self.load_models()
            self.env.render(mode='human')

        mb, mb_obs, mb_ag, mb_dg, mb_action=[],[],[],[],[]
        for i in range(episodes):

            curr_data = self.env.reset()
            while(np.linalg.norm(curr_data['achieved_goal']-curr_data['desired_goal'])<0.05):
                curr_data = self.env.reset()

            observation = curr_data['observation']
            desired_goal = curr_data['desired_goal']
            achieved_goal = curr_data['achieved_goal']
            observation = np.concatenate((observation, desired_goal), axis=0)
            observation_arr, achieved_goal_arr, goal_arr,action_arr=[],[],[],[]
            episode_goal_achieved, episode_observed_acheived= [],[]
            for i in range(50): #guaranteed 50 steps
                action = self.choose_action(observation)
                observation_, _, _, info = self.env.step(action)
                # score += reward

                achieved_goal_ = observation_['achieved_goal']
                observation_ = observation_['observation']
                observation_ = np.concatenate((observation_, desired_goal), axis=0)

                # self.remember(observation, action, reward, observation_, done)
                # if not self.load_checkpoint:
                #     self.learn()
                observation_arr.append(observation)
                action_arr.append(action)
                achieved_goal_arr.append(achieved_goal)
                goal_arr.append(desired_goal)
                observation = observation_
                achieved_goal=achieved_goal_

            achieved_goal_arr.append(achieved_goal)
            observation_arr.append(observation)
            mb_action.append(action_arr)
            mb_ag.append(achieved_goal_arr)
            mb_dg.append(goal_arr)
            mb_obs.append(observation_arr)
        mb_action=np.array(mb_action)
        mb_ag=np.array(mb_ag)
        mb_dg=np.array(mb_dg)
        mb_obs=np.array(mb_obs)
        mb.append([mb_obs,mb_action,mb_dg,mb_ag])
        self.store(mb)

        # if desired--ADD NORMALIZER HERE
        for _ in range(50):
            self.learn()

        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)


        # print("episode", i, "score", score, "average score", avg_score)
        return score_history # _eval_agent--to get score

    def store(self, batches) :
        for batch in batches :
            self.memory.store_transition(batch)
        # we could update NORMALIZER here

    def  learn(self):
        #  must fully load up memory, otherwise must keep learning
        if self.memory.mem_cntr <self.batch_size:
            return
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


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
