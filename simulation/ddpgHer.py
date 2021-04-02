import numpy as np
import sys
import os
from stable_baselines3 import HER
from stable_baselines3.ddpg import DDPG
from stable_baselines3.her import GoalSelectionStrategy
import matplotlib.pyplot as plt
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
        self.model = HER('MlpPolicy', self.env, self.model_class, n_sampled_goal=4,
                         goal_selection_strategy=self.goal_selection_strategy,buffer_size=1000000,
                         batch_size=256, gamma=.95, learning_rate=1e-3, verbose=1, max_episode_length=50)

    def run(self, epochs=5000, train=False):
        # Train the model
        if train:
            # 1000 epochs is approximately 50,000 time steps
            self.model.learn(total_timesteps=(50 * epochs))
            self.model.save("./her_bit_env")

        # WARNING: you must pass an env
        # or wrap your environment with HERGoalEnvWrapper to use the predict method
        self.model = HER.load('./her_bit_env', env=self.env)

        success_rate = []
        for i in range(100):
            obs = self.env.reset()
            score = 0
            success_rate.append(False)
            for j in range(1000):
                action, _ = self.model.predict(obs)

                obs, reward, done, info = self.env.step(action)
                score += reward
                success_rate[-1] = info["is_success"]
                # self.env.render()
                if done:
                    break
                print("epoch: ", j)
                print("score:", score, "average score:", score / j)
            print("success rate: ", success_rate.count(True) / len(success_rate))
        self.plot_success(success_rate, 2)

    def plot_success(self, success_rate, plot_num):
        average = []
        for i, point in enumerate(success_rate):
            average.append(success_rate[:i + 1].count(True) / (i + 1))
        plt.plot(success_rate, color='blue', label="Epoch Success Rate")
        plt.plot(average, color='red', label="Average Success Rate", zorder=3)
        plt.legend()
        plt.title("Success Rate for Simulated FetchReach")
        plt.ylabel("Success Rate")
        plt.xlabel("Number iterations")
        plt.savefig("./plots/success/success_rate_{}.png".format(plot_num))
        plt.clf()
