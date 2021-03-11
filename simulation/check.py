import sys
sys.path.insert(0,"../../gym")
sys.path.insert(0,"../../gym/envs/robotics")
import gym

from stable_baselines.common.env_checker import check_env
from gym.envs.robotics.fetch.reach import FetchReachEnv
# It will check your custom environment and output additional warnings if needed
print(gym)
env=FetchReachEnv()
#env=gym.make()
print(isinstance(env, gym.GoalEnv))
check_env(env)
