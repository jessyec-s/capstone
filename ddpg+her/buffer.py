import gym
import numpy as np
import HER
import math
class ReplayBuffer():
    def __init__(self, max_size, batch_size, input_shape,n_actions, desired_size, achieved_size,sample_func):
        self.mem_size = math.floor(max_size / batch_size)
        self.mem_cntr = 0
        self.sample_func=sample_func
        print(n_actions)
        # initialize memory
        self.memory={'obs' : np.zeros((self.mem_size, batch_size + 1, input_shape)),
                'action': np.zeros((self.mem_size, batch_size, n_actions)),
                'desired_goal':  np.zeros((self.mem_size, batch_size, desired_size)),
                'achieved_goal': np.zeros((self.mem_size, batch_size + 1, achieved_size)),
                }

    def store_transition(self,batch):
        # update memory
        # print(batch[0].shape,batch[1].shape,batch[2].shape,batch[3].shape)
        index=self.mem_cntr % self.mem_size
        # maybe get_storage_idx are needed here
        # print(batch[0].shape)
        self.memory['obs'][index]=batch[0]
        self.memory['action'][index]=batch[1]
        self.memory['desired_goal'][index]=batch[2]
        self.memory['achieved_goal'][index]=batch[3]

        self.mem_cntr+=1

    def sample_buffer(self,batch_size):

        # if memsize is bigger, then memory isn't yet full, otherwise it is
        max_memory = min(self.mem_cntr,self.mem_size)
        #pick random integers from 0 to max memory
        batch = np.random.choice(max_memory,batch_size)

        temp_mem={}
        for key in self.memory.keys():
            temp_mem[key] = self.memory[key][:max_memory]
        temp_mem['obs_next']=temp_mem['obs'][:,1:,:]
        temp_mem['ag_next']=temp_mem['achieved_goal'][:,1:,:]
        # print(temp_mem['action'].shape[1],temp_mem['action'].shape[0])

        transitions= self.sample_func(temp_mem,batch_size)
        # print(batch_size)
        # get HER states here
        return transitions
