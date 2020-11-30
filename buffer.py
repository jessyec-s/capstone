import gym
import numpy as np
import HER
class ReplayBuffer():
    def __init__(self, max_size, batch_size, input_shape,n_actions, desired_size, achieved_size):
        self.mem_size = max_size / batch_size
        self.mem_cntr = 0
        # initialize memory
        self.obs_memory = np.zeros((self.mem_size, batch_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, batch_size, n_actions))
        self.desired_goal_memory = np.zeros(self.mem_size, batch_size, desired_size)
        self.achieved_goal_memory = np.zeros(self.mem_size, batch_size, achieved_size)

    def store_transition(self,batch):
        # update memory
        index=self.mem_cntr % self.mem_size
        self.obs_memory[index]=batch[0]
        self.action_memory[index]=batch[1]
        self.desired_goal_memory[index]=batch[2]
        self.achieved_goal[index]=batch[3]

        self.mem_cntr+=1

    def sample_buffer(self,batch_size):
        # if memsize is bigger, then memory isn't yet full, otherwise it is
        max_memory = min(self.mem_cntr,self.mem_size)
        #pick random integers from 0 to max memory
        batch = np.random.choice(max_memory,batch_size)

        #take those samples for analysis in the state,next state action reward and dones
        states=self.obs_memory[batch] # action and goal concatted
        states_=self.new_state_memory[batch] # action and goal concatted
        actions=self.action_memory[batch]
        rewards=self.reward_memory[batch]
        dones=self.terminal_memory[batch]
        # get HER states here
        #states, states_,actions,rewards,dones=  HER.HERBuff(batch)
        return states, actions, rewards, states_, dones
