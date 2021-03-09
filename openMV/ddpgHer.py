from stable_baselines import HER
from stable_baselines.ddpg import DDPG
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
# from stable_baselines.common.bit_flipping_env import BitFlippingEnv
import numpy as np

class DDPG_HER:
    def __init__(self, env, model_class=DDPG):
        self.model_class = model_class  # works also with SAC, DDPG and TD3
        # Available strategies (cf paper): future, final, episode, random
        self.env = env
        self.goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE
        self.model = HER('MlpPolicy', HERGoalEnvWrapper(self.env), self.model_class, n_sampled_goal=4, goal_selection_strategy=self.goal_selection_strategy, verbose=1)

    def run(self):
        obs = self.env.get_observation()
        print("OBS: ", obs)
        # print("np.array(obs).shape: ", obs.shape)
        print("observation_space: ", self.env.observation_space)
        # Train the model
        self.model.learn(1000)

        self.model.save("./her_bit_env")

        # WARNING: you must pass an env
        # or wrap your environment with HERGoalEnvWrapper to use the predict method
        self.model = HER.load('./her_bit_env', env=self.env)

        # obs = self.env.reset()
        obs = self.env.get_observation()
        print("OBS: ", obs)
        for _ in range(100):
            action, _ = self.model.predict(obs)

            obs, reward, done, _ = self.env.step(action)

            if done:
                return
                # obs = self.env.reset()

