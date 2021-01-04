import numpy as np

class HERBuff():
    def __init__(self,replay_k=8,reward_func=None):
        self.replay_k=replay_k
        self.future_p= 1- 1./(1+replay_k)
        self.reward_func=reward_func # environment reward function

    def sample_her_transitions(self,batch,batch_size_in_transitions):
        # print(batch['action'].shape)
        steps_per_slice = batch['action'].shape[1]
        num_episodes = batch['action'].shape[0]
        batch_size = batch_size_in_transitions

        random_existing_episode_idxs=np.random.randint(0,num_episodes,batch_size)
        tsamp=np.random.randint(steps_per_slice,size=batch_size)

        transitions = {key: batch[key][random_existing_episode_idxs,tsamp].copy() for key in batch.keys()}
        # figure the next couple rows out!!!
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p) # throw out 1/kths of indices
        future_offset = (np.random.uniform(size=batch_size) * (steps_per_slice - tsamp)).astype(int)
        future_t = (tsamp + 1 + future_offset)[her_indexes]

        future_ag = batch['achieved_goal'][random_existing_episode_idxs[her_indexes],future_t]
        transitions['desired_goal'][her_indexes]= future_ag

        transitions['reward']= np.expand_dims(self.reward_func(transitions['ag_next'],transitions['desired_goal'],None),1)
        transitions = { k: transitions[k].reshape(batch_size,*transitions[k].shape[1:]) for k in transitions.keys()}
        # print(transitions['action'].shape)
        return transitions
