import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
import matplotlib as plt
from taxirender import plot_error_drop

def main():
    env=gym.make("InvertedPendulumBulletEnv-v0")
    agent = Agent(input_dims=env.observation_space.shape,env=env,n_actions=env.action_space.shape[0])
    episodes = 250
    filename= 'MoutainCarContinuous.png'
    figure_file= 'plots/'+filename

    best_score = env.reward_range[0]
    score_history=[]
    load_checkpoint= True

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(episodes):
        observation = env.reset()
        # print(type(observation))
        done = False
        score = 0
        while not done:
            action=agent.choose_action(observation)
            # print(action)
            observation_ , reward,done, info = env.step(action)
            # print("reward",reward, "observe", observation)
            score+=reward
            agent.remember(observation,action, reward,observation_,done)
            if not load_checkpoint:
                agent.learn()
            else:
                env.render()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score> best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print("episode",i,"score",score, "average score", avg_score)
    if not load_checkpoint:
        x=[i+1 for i in range(episodes)]
        plot_error_drop(x,score_history)

main()