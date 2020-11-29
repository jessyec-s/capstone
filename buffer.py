import gym
import numpy as np
from copy import deepcopy as dc

class ReplayBuffer():
    def __init__(self, max_size, input_shape,n_actions,k_future):
        self.mem_size = max_size
        self.mem_cntr = 0
        # initialize memory
        self.state_memory = np.zeros((self.mem_size,*input_shape))
        self.next_state_memory = np.zeros((self.mem_size,*input_shape))
        self.action_memory = np.zeros((self.mem_size,n_actions))
        self.reward_memory = np.zeros((self.mem_size))
        self.terminal_memory = np.zeros((self.mem_size,dtype=np.bool))

        # added for her 
        self.achieved_goal_memory = np.zeros((self.mem_size, *input_shape))
        self.desired_goal_memory = np.zeros((self.mem_size, *input_shape))
        self.next_achieved_goal_memory = np.zeros((self.mem_size,*input_shape))
        self.future_p = 1 - (1. / (1 + k_future))


    def store_transition(self,state,action,reward,state_,done, achieved_goal, desired_goal, next_goal):
        # update memory
        # index=self.mem_cntr %self.mem_size
        batch_size=state.shape[0]
        index = self._get_storage_idx(inc=batch_size)
        self.state_memory[index]=state
        self.next_state_memory[index] = state_
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.terminal_memory[index]=done

        self.achieved_goal_memory[index]=achieved_goal
        self.desired_goal_memory[index]=desired_goal
        self.next_achieved_goal_memory[index]=next_goal
        self.mem_cntr+=1

    def _get_storage_idx(self, inc=1):
        """Returns to you the indexes where you will write in the buffer.
        These are consecutive until you hit the end, then they are random.
        """
        if self.mem_cntr+inc <= self.mem_size:
            idx = np.arange(self.mem_cntr, self.mem_cntr+inc)
        elif self.mem_cntr < self.mem_size:
            overflow = inc - (self.mem_size - self.mem_cntr)
            idx_a = np.arange(self.current_size, self.mem_size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.mem_size, inc)

        self.current_size = min(self.mem_size, self.current_size+inc)
        return idx

    def sample_buffer(self,batch_size):
        # if memsize is bigger, then memory isn't yet full, otherwise it is
        max_memory = min(self.mem_cntr,self.mem_size)

        #pick random integers from 0 to max memory
        ep_indices = np.random.choice(max_memory,batch_size)
        time_indices = np.random.choice(max_memory, batch_size)

        states, states_, actions, rewards, dones, desired_goals, next_achieved_goals = []

        for episode, timestep in zip(ep_indices, time_indices):
            #take those samples for analysis in the state,next state action reward and dones
            states.append(dc(self.state_memory[episode][timestep]))
            states_.append(dc(self.new_state_memory[episode][timestep]))
            actions.append(dc(self.action_memory[episode][timestep]))
            # rewards.append(dc(self.reward_memory[episode][timestep]))
            # dones.append(dc(self.terminal_memory[episode][timestep]))
            desired_goals.append(dc(self.desired_goal_memory[episode][timestep]))
            next_achieved_goals.append(dc(self.next_achieved_goal_memory[episode][timestep]))

        states  = np.vstack(states)
        states_ = np.vstack(states_)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)

        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (len(self.next_state_memory) - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + 1 + future_offset)[her_indices]

        future_ag = []
        for episode, f_offset in zip(ep_indices[her_indices], future_t):
            future_ag.append(dc(self.achieved_goal_memory[episode][f_offset]))
        future_ag = np.vstack(future_ag)

        desired_goals[her_indices] = future_ag
        rewards = np.expand_dims(self.env.compute_reward(next_achieved_goals, desired_goals, None), 1)

        # might need to clip our states ?
        return states, actions, rewards, states_, dones, desired_goals
