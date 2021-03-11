import numpy as np
import sys
import os
from stable_baselines3 import HER
from stable_baselines3.ddpg import DDPG
from stable_baselines3.her import GoalSelectionStrategy
# from stable_baselines.common.bit_flipping_env import BitFlippingEnv
import numpy as np
import gym
from gym.envs.robotics.fetch.reach import FetchReachEnv
class DDPG_HER:
    def __init__(self, env, model_class=DDPG):
        self.model_class = model_class  # works also with SAC, DDPG and TD3
        # Available strategies (cf paper): future, final, episode, random
        self.env = env
        self.goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE
        self.model = HER('MlpPolicy', self.env, self.model_class, n_sampled_goal=4, goal_selection_strategy=self.goal_selection_strategy,buffer_size=1000000,batch_size=256,gamma=.95,random_exploration=.3,actor_lr=1e-3, critic_lr=1e-3, noise_type='normal', noise_std=.2, normalize_observations=True, normalize_returns=False, verbose=1,max_episode_length=50)

    def run(self,epochs=500,load_checkpoints=False):
        #obs = self.env.get_observation()
        #print("OBS: ", obs)
        # print("np.array(obs).shape: ", obs.shape)
        print("observation_space: ", self.env.observation_space)
        # Train the model
        self.model.learn(total_timesteps=5000000)
        self.model.save("./her_bit_env")

        # WARNING: you must pass an env
        # or wrap your environment with HERGoalEnvWrapper to use the predict method
        self.model = HER.load('./her_bit_env', env=self.env)

        obs = self.env.reset()
        
        print("OBS: ", obs)
        score=0
        for i in range(1):
            for _ in range(1000):
                action, _ = self.model.predict(obs)

                obs, reward, done, info = self.env.step(action)
                score+=reward
                success_rate.append(info["is_success"])
                self.env.render()
                if done:
                    return
            print("epoch: ",i) 
        print("score:", score, "average score:", average_score)
        

