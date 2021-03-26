from stable_baselines3 import HER
from stable_baselines3.ddpg import DDPG
import time

class DDPG_HER:
    def __init__(self, env, model_class=DDPG):
        self.model_class = model_class  # works also with SAC, DDPG and TD3
        # Available strategies (cf paper): future, final, episode, random
        self.env = env
        self.goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE
        self.model = HER('MlpPolicy', self.env, self.model_class, n_sampled_goal=4, goal_selection_strategy=self.goal_selection_strategy,
                        buffer_size = 1000000, batch_size = 256, gamma = .95, learning_rate = 1e-3, verbose=1,  max_episode_length=50)

    def run(self, train_epochs=5000, train=False):
        # print("np.array(obs).shape: ", obs.shape)
        print("observation_space: ", self.env.observation_space)
        # Train the model
        if train:
            # 1000 epochs is approximately 50,000 time steps
            self.model.learn(total_timesteps=(50 * train_epochs))
            self.model.save("./her_bit_env")

        # WARNING: you must pass an env
        # or wrap your environment with HERGoalEnvWrapper to use the predict method
        # TODO: convert the loaded data to the proper dimensions
        self.model = HER.load('./her_bit_env', env=self.env)

        obs = self.env.get_observation_simulated()

        for i in range(1):
            obs = self.env.reset()
            score = 0
            self.env.success_history.append(False)
            for j in range(1000):
                # obs needs simulated coords
                action, _ = self.model.predict(obs)

                obs, reward, done, info = self.env.step(action)
                score += reward

                self.env.success_history[-1] = done
                print("Distance history: ", self.env.distance_history[-1])

                if done:
                    break
                time.sleep(1)
                print("epoch: ", j)
                print("score:", score, "average score:", score / j)
            print("success rate: ", self.env.success_history.count(True) / len(self.env.success_history))

        return self.env.success_hsitory, self.env.distance_history

