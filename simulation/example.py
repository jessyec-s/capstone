import sys
sys.path.insert(0,"../../gym")
sys.path.insert(0,"../../gym/envs/robotics")
import gym 
from gym.envs.robotics.fetch.reach import FetchReachEnv
#from fetch.reach import FetchReachEnv

print(gym)
#env = FetchPickAndPlaceEnv()
env = FetchReachEnv()
env.reset()
#for _ in range(10000):
env.render()
    #print(f"action:{env.action_space.sample()}")
#    action = env.action_space.sample() # take a random action
#    observation, reward, done, info = env.step(action)
    
    
    
#env.close()
