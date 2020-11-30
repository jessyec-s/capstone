import pybullet_envs
import gym
import numpy as np
from ddpg_torch import Agent
import matplotlib as plt

def main():

    actor_learning_rate=[1*10**-4, 3*10**-4, 6*10**-4, 10**-3, 3*10**-3, 6*10**-3, 10**-2]
    critic_learning_rate=[1*10**-4, 3*10**-4, 6*10**-4, 10**-3, 3*10**-3, 6*10**-3, 10**-2]
    tau=[.9,.93,.95,.97,.99]
    batch_size=[32,64,128,256]
    p_rand=[0,.1,.2,.3,.4]
    sigma=[0,.1,.2,.3,.4]
    L2_norm_coeff=[0,.01,.03,.1,.3,.6,1]

    load_checkpoint=False

    env=gym.make("FetchReach-v1")
    env_dims = env.reset()['observation'].shape[0] + env.reset()['desired_goal'].shape[0]
    agent = Agent(input_dims=env_dims,n_actions=env.action_space.shape[0])
    episodes = 250
    filename= 'FetchAndReach.png'
    figure_file= 'plots/'+filename

    best_score = env.reward_range[0]
    score_history=[]

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(episodes):
        curr_data = env.reset()
        done = False
        score = 0
        while not done:
            observation = curr_data['observation']
            desired_goal = curr_data['desired_goal']
            achieved_goal = curr_data['achieved_goal']
            observation = np.concatenate((observation, desired_goal), axis=0)
            action=agent.choose_action(observation)
            observation_ , reward,done, info = env.step(action)
            score+=reward

            observation_ = observation_['observation']
            desired_goal_ = curr_data['desired_goal']
            observation_ = np.concatenate((observation_, desired_goal_), axis=0)

            agent.remember(observation, action, reward, observation_, done)
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

def plot_error_drop(tot_penalties, tot_epochs):
    line1 = mpl.plot(np.arange(0, len(tot_penalties)), tot_penalties, 'b', label="penalties")
    line2 = mpl.plot(np.arange(0, len(tot_epochs)), tot_epochs, 'r', label='epochs')
    mpl.ylabel("final value")
    mpl.xlabel("episode")
    mpl.title("error drop over time")
    mpl.legend(loc="upper right")
    mpl.show()

main()
