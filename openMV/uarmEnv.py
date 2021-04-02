import numpy as np
import gym
from gym import spaces
import math

MIN = 0
MAX = 1
RADIUS_LIMIT = (150., 250.)
ANGLE_LIMIT = (0., 180.)
HEIGHT_LIMIT = (0., 150.)
X_OFFSET = 0.8
Y_OFFSET = 0.75

def convertToPolar(x, y, z):
    return np.array([math.sqrt(x**2 + y**2), math.atan(y/x)*(180/math.pi), z])

def convertToCartesian(r, theta, z):
    theta = theta * (math.pi/180)
    return np.array([r*math.cos(theta), r*math.sin(theta), z])

def convertToCartesianRobot(r, theta, z):
    theta = theta * (math.pi/180)
    # Need to reverse x and y due to the way it is laid out in the physical system coordinate system
    return np.array([r*math.sin(theta), -r*math.cos(theta), z])

class UarmEnv(gym.GoalEnv):
    def __init__(self, uarm_controller, timeout=None, **kwargs):
        super(UarmEnv, self).__init__()
        self.MULT_FACTOR_SIM = 0.05
        self.MULT_FACTOR_R = 9.615
        self.MULT_FACTOR_Z = 11.09
        self.distance_threshold = 15.0
        self.uarm_controller = uarm_controller
        self.distance_history = []
        self.success_history = []
        self.time_history = []

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
        self.distance_history = []

        self.suction_state = np.zeros(1)
        print("RESET RETURNS: ", self.get_observation_simulated())

        return self.get_observation_simulated()  # reward, done, info can't be included

    # Old reward function- based only on distance calc
    # def compute_reward(self, achieved_goal, goal):
    #     # Compute distance between goal and the achieved goal.
    #     d = self.goal_distance(achieved_goal, goal)
    #     self.distance_history.append(d)
    #     return -(d > self.distance_threshold).astype(np.float32)

    # New reward method that is based on difference between the coords + some other penalties
    def compute_reward(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        threshold = 11
        reward = 0
        print("achieved goal and desired goal:", achieved_goal, goal)
        print("diff: ", abs(achieved_goal[0] - goal[0]), abs(achieved_goal[1] - goal[1]), abs(achieved_goal[2] - goal[2]))
        reward += (abs(achieved_goal[0] - goal[0]) <= threshold).astype(np.float32)/3
        reward += (abs(achieved_goal[1] - goal[1]) <= threshold).astype(np.float32)/3
        reward += (abs(achieved_goal[2] - goal[2]) <= threshold).astype(np.float32)/3
        d = self.goal_distance(achieved_goal, goal)
        if len(self.distance_history) > 0:
            if d >= self.distance_history[-1]:
                if (abs(d-self.distance_history[-1]) > 100):
                    reward -= 1
                else:
                    reward -= 0.1
            else:
                reward += 0.1
                if d < 20:
                    reward += 0.1
        print("reward: ", reward)
        self.distance_history.append(d)
        return reward

    # New is_success function based on object coords
    def is_success(self, achieved_goal, desired_goal):
        # Compute distance between goal and the achieved goal.
        reward_threshold = 0.1
        threshold = 10
        reward = 0
        reward += (abs(achieved_goal[0] - desired_goal[0]) <= threshold).astype(np.float32)/3
        reward += (abs(achieved_goal[1] - desired_goal[1]) <= threshold).astype(np.float32)/3
        reward += (abs(achieved_goal[2] - desired_goal[2]) <= threshold).astype(np.float32)/3
        d = self.goal_distance(achieved_goal, desired_goal)
        print("is success reward: ", reward)
        return (abs(reward - 1) <= reward_threshold or (d < self.distance_threshold).astype(np.float32))

    # Old is_success function based on distance
    # def is_success(self, achieved_goal, desired_goal):
    #     d = self.goal_distance(achieved_goal, desired_goal)
    #     return (d < self.distance_threshold).astype(np.float32)

    def step(self, action):
        lastPos = self.uarm_controller.get_position()
        # clip simulated action at maximum positions
        print("last pos x, y, z: ", lastPos)
        print("ACTION start x, y, z: ", action)

        action = np.clip(action, -1, 1)

        # Scale simulated action
        action[0] *= self.MULT_FACTOR_R
        action[1] *= self.MULT_FACTOR_R
        action[2] *= self.MULT_FACTOR_Z
        print("ACTION after scale x, y, z: ", action)

        # Add action and last position
        new_pos = [sum(x) for x in zip(lastPos, action)]

        # Convert new pos to polar
        new_pos = convertToPolar(new_pos[0], new_pos[1], new_pos[2])
        # To convert physical system default polar axis to standard
        new_pos[1] += 90.

        # Clip physical bounds
        if not self.uarm_controller.check_pos_is_limit(new_pos.tolist(), is_polar=True):
            new_pos[0] = np.clip(new_pos[0], RADIUS_LIMIT[MIN], RADIUS_LIMIT[MAX])
            new_pos[1] = np.clip(new_pos[1], ANGLE_LIMIT[MIN], ANGLE_LIMIT[MAX])
            new_pos[2] = np.clip(new_pos[2], HEIGHT_LIMIT[MIN], HEIGHT_LIMIT[MAX])

        print("new pos polar: ", new_pos)

        # if last position and new position are different, move the robot
        lastPos = self.uarm_controller.get_polar()
        print("last pos polar: ", lastPos)

        if not np.array_equal(lastPos, new_pos):
            self.uarm_controller.set_polar(stretch=new_pos[0], rotation=new_pos[1], height=new_pos[2], wait=True)

        desired_goal = self.object_pos
        achieved_goal = self.uarm_controller.get_position()
        print("desired goal: ", desired_goal)
        print("achieved goal: ", achieved_goal)

        simulated_obs = self.get_observation_simulated()

        print("simulated_obs after offset: ", simulated_obs)

        return simulated_obs, self.compute_reward(achieved_goal, desired_goal), self.is_success(achieved_goal, desired_goal), {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # radius: 150-250 vs 0.2 -> 0.72
    # theta: 0-180 vs -90 -> +90
    # z: 0-150 vs 0.324 -> 1
    # Simulated to Phys
    def get_obs_sim_to_phys(self, x):
        print("In obs sim to phys")
        obs = x
        print("obs before: ", obs)
        obs[0]=(obs[0]-.2)*(250-150)/(.72-.2)+150
        obs[1]=(obs[1]+90)
        obs[2]=(obs[2]-.324)*(150)/(1.0-.324)
        print("obs after: ", obs)
        return obs

    # Physical to Simulated
    def get_obs_phys_to_sim(self, x):
        # Reverse
        # Phys ploar: [216.80258237229708, 89.9080001961666, 46.718538609349665]
        # phys converted: [0.79914021, 0.20264068, 0.53455147]
        # sim: [1.34737272, 0.74912108, 0.53454488]
        obs = x
        obs[0] = (obs[0] - 150) * (.72 - .2) / (250 - 150) + 0.2
        obs[1] = (obs[1] - 90)
        obs[2] = (obs[2]) * (1.0 - .324) / (150) + .324
        return obs

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
        obj_pos_polar[1] += 90
        print("desired goal in phys system: ", obj_pos_polar)
        obj_pos_polar = self.get_obs_phys_to_sim(obj_pos_polar)

        print("Current position in get_obs_simulated: ", self.uarm_controller.get_polar())
        current_position = self.get_obs_phys_to_sim(np.array(self.uarm_controller.get_polar()))

        # convert all to cartesian for use in ddpg
        # obj_pos_polar[1] -= 90
        # current_position[1] -= 90
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

        observation = self.addSimulatedOffset(observation)

        return observation

    def addSimulatedOffset(self, simulated_obs):
        simulated_obs['observation'][0] += X_OFFSET
        simulated_obs['observation'][1] += Y_OFFSET
        simulated_obs['achieved_goal'][0] += X_OFFSET
        simulated_obs['achieved_goal'][1] += Y_OFFSET
        simulated_obs['desired_goal'][0] += X_OFFSET
        simulated_obs['desired_goal'][1] += Y_OFFSET
        return simulated_obs

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
