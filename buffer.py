import gym
import numpy as np
import HER
class ReplayBuffer():
    def __init__(self, max_size, input_shape,n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        # initialize memory
        self.state_memory = np.zeros((self.mem_size,input_shape))
        self.new_state_memory = np.zeros((self.mem_size,input_shape))
        self.action_memory = np.zeros((self.mem_size,n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool)

    def store_transition(self,state,action,reward,state_,done):
        # update memory
        index=self.mem_cntr %self.mem_size
        self.state_memory[index]=state
        self.new_state_memory[index] = state_
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.terminal_memory[index]=done
        self.mem_cntr+=1

    def sample_buffer(self,batch_size):
        # if memsize is bigger, then memory isn't yet full, otherwise it is
        max_memory = min(self.mem_cntr,self.mem_size)
        #pick random integers from 0 to max memory
        batch = np.random.choice(max_memory,batch_size)

        #take those samples for analysis in the state,next state action reward and dones
        states=self.state_memory[batch] # action and goal concatted
        states_=self.new_state_memory[batch] # action and goal concatted
        actions=self.action_memory[batch]
        rewards=self.reward_memory[batch]
        dones=self.terminal_memory[batch]
        # get HER states here
        #states, states_,actions,rewards,dones=  HER.HERBuff(batch)
        return states, actions, rewards, states_, dones
