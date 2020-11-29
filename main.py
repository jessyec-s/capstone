import pybullet_envs
import gym
import numpy as np
from ddpg_torch import Agent
import matplotlib as plt

def eval_agent(env_, agent_):
    total_success_rate = []
    running_r = []
    for ep in range(10):
        per_success_rate = []
        env_dictionary = env_.reset()
        s = env_dictionary["observation"]
        ag = env_dictionary["achieved_goal"]
        g = env_dictionary["desired_goal"]
        while np.linalg.norm(ag - g) <= 0.05:
            env_dictionary = env_.reset()
            s = env_dictionary["observation"]
            ag = env_dictionary["achieved_goal"]
            g = env_dictionary["desired_goal"]
        ep_r = 0
        for t in range(50):
            with torch.no_grad():
                a = agent_.choose_action(s, g, train_mode=False)
            observation_new, r, _, info_ = env_.step(a)
            s = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info_['is_success'])
            ep_r += r
        total_success_rate.append(per_success_rate)
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, ep_r

def main():

    actor_learning_rate=[1*10**-4, 3*10**-4, 6*10**-4, 10**-3, 3*10**-3, 6*10**-3, 10**-2]
    critic_learning_rate=[1*10**-4, 3*10**-4, 6*10**-4, 10**-3, 3*10**-3, 6*10**-3, 10**-2]
    tau=[.9,.93,.95,.97,.99]
    batch_size=[32,64,128,256]
    p_rand=[0,.1,.2,.3,.4]
    sigma=[0,.1,.2,.3,.4]
    L2_norm_coeff=[0,.01,.03,.1,.3,.6,1]

    load_checkpoint=True

    env=gym.make("CartPoleContinuousBulletEnv-v0")

    agent = Agent(input_dims=env.observation_space.shape,n_actions=env.action_space.shape[0])
    episodes = 2
    epochs = 50
    cycles = 10
    updates = 40
    k_future = 4
    filename= 'MoutainCarContinuous.png'
    figure_file= 'plots/'+filename

    best_score = env.reward_range[0]
    score_history=[]

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    total_success_rate = []
    total_actor_loss = []
    total_critic_loss = []
    for epoch in range(epochs):
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        for cycle in range(cycles):
            mb = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
            for i in range(episodes):
                episode_dict = {
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []
                }
                env_dict = env.reset()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                    env_dict = env.reset()
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]

                # done = False
                # score = 0
                for t in range(50):
                # while not done:
                    action=agent.choose_action(state, desired_goal) # TODO: update choose action 
                    nex_env_dict , reward,done, info = env.step(action)

                    next_state = env_dict["observation"]
                    next_achieved_goal = env_dict["achieved_goal"]
                    next_desired_goal = env_dict["desired_goal"]

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action.copy())
                    episode_dict["achieved_goal"].append(achieved_goal.copy())
                    episode_dict["desired_goal"].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()

                    # score+=reward

                episode_dict["state"].append(state.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                mb.append(dc(episode_dict))

            agent.remember(state,action, reward,next_env_dict,done) # TODO: update remember
            for n_update in range(updates):
                actor_loss, critic_loss = agent.learn()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss /num_updates
            agent.update_networks()

            

                if not load_checkpoint:
                    agent.learn()
                else:
                    env.render()
                state = next_env_dict

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