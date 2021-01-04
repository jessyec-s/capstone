import pybullet_envs
import gym
import numpy as np
from ddpg_torch import Agent
import matplotlib.pyplot as mpl

def main():

    actor_learning_rate=[1*10**-4, 3*10**-4, 6*10**-4, 10**-3, 3*10**-3, 6*10**-3, 10**-2]
    critic_learning_rate=[1*10**-4, 3*10**-4, 6*10**-4, 10**-3, 3*10**-3, 6*10**-3, 10**-2]
    tau=[.9,.93,.95,.97,.99]
    batch_size=[32,64,128,256]
    p_rand=[0,.1,.2,.3,.4]
    sigma=[0,.1,.2,.3,.4]
    L2_norm_coeff=[0,.01,.03,.1,.3,.6,1]

    load_checkpoint=True
    epochs=40
    env=gym.make("FetchReach-v1")
    agent = Agent(n_actions=env.action_space.shape[0],load_checkpoint=load_checkpoint,env=env,epochs=epochs)
    if load_checkpoint is False:
        score_history=agent.train()
    else:
        agent.load_models()
        agent.env.render(mode='human')
        agent.episodes=40
        score_history=agent.eval_agent()

    if not load_checkpoint:
        x=[i+1 for i in range(epochs)]
        plot_error_drop(score_history)

def plot_error_drop(tot_penalties, tot_epochs):
    # line1 = mpl.plot(np.arange(0, len(tot_penalties)), tot_penalties, 'b', label="penalties")
    line2 = mpl.plot(np.arange(0, len(tot_epochs)), tot_epochs, 'r', label='epochs')
    mpl.ylabel("final value")
    # mpl.xlabel("episode")
    mpl.title("error drop over time")
    mpl.legend(loc="upper right")
    mpl.show()

main()
