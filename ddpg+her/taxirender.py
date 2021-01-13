import gym
from time import sleep
from IPython.display import clear_output
import random
import numpy as np
import matplotlib.pyplot as mpl


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print("Timestep: {i + 1}")
        print("State: {frame['state']}")
        print("Action: {frame['action']}")
        print("Reward: {frame['reward']}")
        sleep(.1)


def reinforcementAlg(env, alpha, gamma, episodes):
    #initialize Q space
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    tot_penalties = []
    tot_epochs = []
    #run episodes of path finder
    for i in range(episodes):
        frames = []  # for animation
        env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False
        while not done:
            prev_state = env.s
            # get best action for the given state
            max_actions = [i for i, x in enumerate(Q[env.s]) if x == Q[prev_state].max()]
            random.shuffle(max_actions)
            selected_action = max_actions.pop()

            # take the state, and choose the reward
            state, reward, done, info = env.step(selected_action)
            #update your Q function
            Q[prev_state][selected_action] = (1 - alpha) * Q[prev_state][selected_action] + alpha * (
                    reward + gamma * Q[state].max())

            penalties += -reward

            # Put each rendered frame into dict for animation
            #for visualizing state info
            frames.append({
                'frame': env.render(mode='ansi'), #print out this for the state map
                'state': state,
                'action': selected_action,
                'reward': reward
            }
            )

            epochs += 1
        # just for debug purposes
        print('episode: ',i,"epochs: ",epochs, 'penalties: ',penalties)
        # if i == 400:
            # print_frames(frames)
        tot_penalties.append(penalties)
        tot_epochs.append(epochs)

    return tot_penalties, tot_epochs


#calling function
def taxi_render():
    env = gym.make("CartPole-v0").env
    env.render()
    print(env.observation_space.high,env.observation_space.low)
    alpha = .9
    gamma = .1
    episodes = 1
    # djh
    # tot_penalties, tot_epochs = reinforcementAlg(env, alpha, gamma, episodes)
    # plot_error_drop(tot_penalties, tot_epochs)

