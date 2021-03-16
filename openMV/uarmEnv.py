import numpy as np
import gym
from gym import spaces
import math

MIN = 0
MAX = 1
RADIUS_LIMIT = (150., 250.)
ANGLE_LIMIT = (0., 180.)
HEIGHT_LIMIT = (0., 150.)

def convertToPolar(x, y, z):
    return np.array([math.sqrt(x**2 + y**2), math.atan(y/x)*(180/math.pi), z])

def convertToCartesian(r, theta, z):
    theta = theta * (math.pi/180)
    return np.array([r*math.cos(theta), r*math.sin(theta), z])

class UarmEnv(gym.GoalEnv):
    def __init__(self, uarm_controller, timeout=None, **kwargs):
        super(UarmEnv, self).__init__()
        self.MULT_FACTOR_SIM = 0.05
        self.MULT_FACTOR_R = 9.615
        self.MULT_FACTOR_Z = 11.09
        self.distance_threshold = 0.05
        self.uarm_controller = uarm_controller
        self.distance_history = []

        self.object_pos = np.zeros(3)  # in cartesian coordinates
        self.object_rel_pos = np.zeros(3)
        self.suction_state = np.zeros(1)
        self.suction_pos = np.zeros(3)

        # Define action and observation space
        # TODO: not sure if the shape here is the right value
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(10,), dtype='float32'),
        ))

    def reset(self):
        self.uarm_controller.UArm_reset(True)
        self.uarm_controller.do_suction(False)
        self.suction_state = False
        # self.object_pos = np.zeros(3)  # in cartesian coordinates
        # self.update_object_rel(curr_pos)

        self.suction_state = np.zeros(1)
        print("RESET RETURNS: ", self.get_observation_simulated())

        return self.get_observation_simulated()  # reward, done, info can't be included

    def compute_reward(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        self.distance_history.append(d)
        return -(d > self.distance_threshold).astype(np.float32)

    def is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def step(self, action):
        lastPos = self.uarm_controller.get_position()
        # u = self.uarm_controller.coordinate_to_angles(u)
        # clip simulated action at maximum positions
        print("last pos: ", lastPos)
        print("ACTION start: ", action)

        action = np.clip(action, -1, 1)

        # Scale simulated action
        action[0] *= self.MULT_FACTOR_R
        action[1] *= self.MULT_FACTOR_R
        action[2] *= self.MULT_FACTOR_Z
        print("ACTION after scale: ", action)

        # Add action and last position
        new_pos = [sum(x) for x in zip(lastPos, action)]

        # Convert new pos to polar
        new_pos = convertToPolar(new_pos[0], new_pos[1], new_pos[2])

        # Clip physical bounds
        if not self.uarm_controller.check_pos_is_limit(new_pos.tolist(), is_polar=True):
            new_pos[0] = np.clip(new_pos[0], RADIUS_LIMIT[MIN], RADIUS_LIMIT[MAX])
            new_pos[1] = np.clip(new_pos[1], ANGLE_LIMIT[MIN], ANGLE_LIMIT[MAX])
            new_pos[2] = np.clip(new_pos[2], HEIGHT_LIMIT[MIN], HEIGHT_LIMIT[MAX])

        print("new pos: ", new_pos)

        # if last position and new position are different, move the robot
        lastPos = self.uarm_controller.get_polar()
        if not np.array_equal(lastPos, new_pos):
            self.uarm_controller.set_polar(stretch=new_pos[0], rotation=new_pos[1], height=new_pos[2], wait=True)
        # self.update_object_rel(self.uarm_controller.get_position())

        desired_goal = self.object_pos
        achieved_goal = self.uarm_controller.get_position()
        print("desired goal: ", desired_goal)
        print("achieved goal: ", achieved_goal)

        return self.get_observation_simulated(), self.compute_reward(achieved_goal, desired_goal), self.is_success(achieved_goal, desired_goal), {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # radius: 150-250 vs 0.2 -> 0.72
    # theta: 0-180 vs -90 -> +90
    # z: 0-150 vs 0.324 -> 1
    # Physical to Simulated
    def get_obs_phys_to_sim(self, obs):
        obs[0]=(obs[0]-.2)*(250-150)/(.72-.2)+150
        obs[1]=(obs[1]+90)
        obs[2]=(obs[2]-.324)*(150)/(1.0-.324)
        return obs

    # def set_obs_phys_to_sim(self, x, polar=True):
    #     obs = x
    #     obs[0]=(obs[0]-150)*(.72-.2)/(250-150)+.2
    #     obs[1]=obs[1]-90
    #     obs[2]=(obs[2])*(1.0-.324)/(150)+.324
    #     # self.set_observation(obs,polar)
    #     return obs

    # Simulated to Physical
    def get_obs_sim_to_phys(self, x):
        # Reverse
        obs = x
        obs[0] = (obs[0] - .2) * (.72 - .2) / (250 - 150) + 0.2
        obs[1] = (obs[1] + 90)
        obs[2] = (obs[2]) * (1.0 - .324) / (150)
        return obs
    #
    # def set_obs_sim_to_phys(self, obs):
    #     # Reverse
    #
    #     obs[0] = (obs[0] - 0.2) * (250 - 150) / (.72 - .2) + 150
    #     obs[1] = obs[1] + 90
    #     obs[2] = (obs[2]) * (150) / (1.0 - .324)
    # # TODO: Set observation (maybe not needed)
    # self.set_observation(obs,polar)

    # obs = np.concatenate([
    #     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
    #     object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
    # ])

    def get_observation(self):
        current_position = self.get_obs_phys_to_sim(np.array(self.uarm_controller.get_polar()))
        current_position_cartesian = convertToCartesian(current_position[0], current_position[1], current_position[2])

        obs = np.concatenate([
            current_position_cartesian, np.zeros(4),
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': current_position_cartesian,
            'desired_goal': np.array(self.object_pos),
        }

    # Convert simulated coords to physical
    def get_observation_simulated(self):

        obj_pos_polar = convertToPolar(self.object_pos[0], self.object_pos[1], self.object_pos[2])
        obj_pos_polar = self.get_obs_phys_to_sim(obj_pos_polar)

        current_position = self.get_obs_phys_to_sim(np.array(self.uarm_controller.get_polar()))

        # convert all to cartesian for use in ddpg
        obj_pos_cartesian = convertToCartesian(obj_pos_polar[0], obj_pos_polar[1], obj_pos_polar[2])
        current_position_cartesian = convertToCartesian(current_position[0], current_position[1], current_position[2])

        obs = np.concatenate([
            current_position_cartesian, np.zeros(7),
        ])

        observation = {
            'observation': obs.copy(),
            'achieved_goal': current_position_cartesian,
            'desired_goal': obj_pos_cartesian,
        }
        print("observation shape: ", observation['observation'].shape)

        return observation

    def get_object_pos(self):
        return self.object_pos

    def set_object_pos(self, pos):
        # maybe convert to polar
        self.object_pos = pos

    def set_suction_state(self, state):
        self.suction_state = state

    def goal_distance(self, goal_a, goal_b):
        assert len(goal_a) == len(goal_b)
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def update_object_rel(self, uarm_pos):
        # TODO make sure this subtraction is in the right order
        x = uarm_pos[0] - self.object_pos[0]
        y = uarm_pos[1] - self.object_pos[1]
        z = uarm_pos[2] - self.object_pos[2]
        self.object_rel_pos = np.array(self.uarm_controller.coordinate_to_angles(x, y, z))
