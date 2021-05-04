import numpy as np
import gym
from gym import spaces
import math

# UArm Environment Constants
MIN = 0
MAX = 1
RADIUS_LIMIT = (150., 300.)
ANGLE_LIMIT = (0., 180.)
HEIGHT_LIMIT = (0., 150.)
X_OFFSET = 0.8
Y_OFFSET = 0.75

def convertToPolar(x, y, z):
    """
    Converts cartesian coordinates to polar/cylindrical coordinates.

    Parameters:
        x (int): the x coordinate of the Uarm
        y (int): the y coordinate of the Uarm
        z (int): the z coordinate of the Uarm

    Returns:
        [r, theta, z] - numpy array of converted polar/cylindrical coordinates
    """
    return np.array([math.sqrt(x**2 + y**2), math.atan(y/x)*(180/math.pi), z])

def convertToCartesian(r, theta, z):
    """
    Converts polar/cylindrical coordinates to cartesian coordinates.

    Parameters:
        r (int): the radius of the Uarm
        theta (int): the angle of the Uarm
        z (int): the z coordinate of the Uarm

    Returns:
        [x, y, z] - numpy array of converted cartesian coordinates
    """
    theta = theta * (math.pi/180)
    return np.array([r*math.cos(theta), r*math.sin(theta), z])

def convertToCartesianRobot(r, theta, z):
    """
    Converts polar/cylindrical coordinates to cartesian coordinates- used for testing purposes.

    Parameters:
        r (int): the radius of the Uarm
        theta (int): the angle of the Uarm
        z (int): the z coordinate of the Uarm

    Returns:
        [x, y, z] - numpy array of converted cartesian coordinates
    """
    theta = theta * (math.pi/180)
    # Need to reverse x and y due to the way it is laid out in the Uarm coordinate system
    return np.array([r*math.sin(theta), -r*math.cos(theta), z])

class UarmEnv(gym.GoalEnv):
    """
    Class that extends the gym GoalEnv and is used to implement a gym environment for the physical system.
    It is used in the reinforcement learning algorithm, and is responsible for converting between physical and simulated environments.

    Parameters:
        uarm_controller (UarmController): an instance of the UarmController class

    """
    def __init__(self, uarm_controller, **kwargs):
        super(UarmEnv, self).__init__()

        # Conversion constants between physical and simulated systems 
        self.MULT_FACTOR_SIM = 0.05
        self.MULT_FACTOR_R = 9.615
        self.MULT_FACTOR_Z = 11.09

        # Arm distance to the object which is considered a success
        self.distance_threshold = 10.0

        self.uarm_controller = uarm_controller
        self.distance_history = []
        self.success_history = []
        self.time_history = []

        self.object_pos = np.zeros(3)  # in cartesian coordinates
        self.object_rel_pos = np.zeros(3)
        self.suction_state = np.zeros(1)
        self.suction_pos = np.zeros(3)

        # Define action and observation space
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(10,), dtype='float32'),
        ))

    def reset(self):
        """
        Resets the Uarm environment to a default position.

        Returns:
            observation (dict): dictionary containing the observation, achieved_goal, and desired_goal
            
        """
        self.uarm_controller.UArm_reset(True)
        self.uarm_controller.do_suction(False)
        self.distance_history = []
        self.suction_state = np.zeros(1)

        print("RESET RETURNS: ", self.get_observation_simulated())

        return self.get_observation_simulated()

    def compute_reward_old(self, achieved_goal, goal):
        """
        A bare bones reward function which just compares the distance from the arm to the object with the defined success threshold.
        The current model was trained with this reward function.

        Parameters:
            achieved_goal (array): the current position of the Uarm
            goal (array): the position of the object and the desired_goal of the Uarm

        Returns:
            A reward value based on whether the robot was successful or not
            
        """
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        self.distance_history.append(d)
        return -(d > self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal):
        """
        An updated reward function which is based on the difference between the individual coordinates.
        It also includes some reward penalties for sharp changes in distance from one timestep to the next.

        Parameters:
            achieved_goal (array): the current position of the Uarm
            goal (array): the position of the object and the desired_goal of the Uarm

        Returns:
            A reward value based on the robot's distance to the object.
            
        """
        # Compute distance between goal and the achieved goal.
        threshold = 11
        reward = 0
        print("achieved goal and desired goal:", achieved_goal, goal)
        print("diff: ", abs(achieved_goal[0] - goal[0]), abs(achieved_goal[1] - goal[1]), abs(achieved_goal[2] - goal[2]))

        # Incremental reward for each coordinate
        reward += (abs(achieved_goal[0] - goal[0]) <= threshold).astype(np.float32)/3
        reward += (abs(achieved_goal[1] - goal[1]) <= threshold).astype(np.float32)/3
        reward += (abs(achieved_goal[2] - goal[2]) <= threshold).astype(np.float32)/3

        d = self.goal_distance(achieved_goal, goal)

        # Reward penalties
        if len(self.distance_history) > 0:
            # Penalize if distance is getting larger
            if d >= self.distance_history[-1]:
                # Penalize if there is a large change in location from the previous timestep
                if (abs(d-self.distance_history[-1]) > 100):
                    reward -= 1
                else:
                    reward -= 0.1
            else:
                # Additional reward if distance is decreasing each timestep
                reward += 0.1
                if d < 20:
                    reward += 0.1

        print("reward: ", reward)
        self.distance_history.append(d)
        return reward

    # Old is_success function based on distance
    def is_success_old(self, achieved_goal, desired_goal):
        """
        Determines whether the find and touch task was successful.
        The current model was trained with this success function.

        Parameters:
            achieved_goal (array): the current position of the Uarm
            desired_goal (array): the position of the object and the desired_goal of the Uarm

        Returns:
            A boolean value indicating success (True) or Failure (False).
            
        """
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    # New is_success function based on object coords
    def is_success(self, achieved_goal, desired_goal):
        """
        Determines whether the find and touch task was successful by comparing the distance of each individual coordinate to the goal.

        Parameters:
            achieved_goal (array): the current position of the Uarm
            desired_goal (array): the position of the object and the desired_goal of the Uarm

        Returns:
            A boolean value indicating success (True) or Failure (False).
            
        """
        reward_threshold = 0.1
        threshold = 10
        reward = 0

        # Compares each coordinate's distance from the goal
        reward += (abs(achieved_goal[0] - desired_goal[0]) <= threshold).astype(np.float32)/3
        reward += (abs(achieved_goal[1] - desired_goal[1]) <= threshold).astype(np.float32)/3
        reward += (abs(achieved_goal[2] - desired_goal[2]) <= threshold).astype(np.float32)/3
        d = self.goal_distance(achieved_goal, desired_goal)
    
        return (abs(reward - 1) <= reward_threshold or (d < self.distance_threshold).astype(np.float32))

    def step(self, action):
        """
        Used in the RL algorithm to move the robot according to the returned action.
        As the RL algorithm is currently implemented, the action is returned in the coordinate system of the simulated environment,
        so we must convert this action to the physical system environment before we apply it.

        Parameters:
            action (array): the action the Uarm should take.

        Returns:
            obs: observation data array 
            reward: the reward that the robot achieved by taking this step
            done: whether the robot achieved success after this step 
            info: any additional information 
            
        """
        # The current cartesian position of the Uarm
        lastPos = self.uarm_controller.get_position()

        # Clip simulated action at maximum positions
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

        # Convert physical system default polar axis to standard
        new_pos[1] += 90.

        # Clip physical bounds
        if not self.uarm_controller.check_pos_is_limit(new_pos.tolist(), is_polar=True):
            new_pos[0] = np.clip(new_pos[0], RADIUS_LIMIT[MIN], RADIUS_LIMIT[MAX])
            new_pos[1] = np.clip(new_pos[1], ANGLE_LIMIT[MIN], ANGLE_LIMIT[MAX])
            new_pos[2] = np.clip(new_pos[2], HEIGHT_LIMIT[MIN], HEIGHT_LIMIT[MAX])

        lastPos = self.uarm_controller.get_polar()

        # if last position and new position are different, move the robot
        if not np.array_equal(lastPos, new_pos):
            self.uarm_controller.set_polar(stretch=new_pos[0], rotation=new_pos[1], height=new_pos[2], wait=True)

        # Set new achieved_goal
        desired_goal = self.object_pos
        achieved_goal = self.uarm_controller.get_position()
        print("desired goal: ", desired_goal)
        print("achieved goal: ", achieved_goal)

        simulated_obs = self.get_observation_simulated()

        return simulated_obs, self.compute_reward(achieved_goal, desired_goal), self.is_success(achieved_goal, desired_goal), {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # radius: 150-250 vs 0.2 -> 0.72
    # theta: 0-180 vs -90 -> +90
    # z: 0-150 vs 0.324 -> 1
    def get_obs_sim_to_phys(self, obs):
        """
        Convert positional observation parameters from the simulated environment to the physical system environment.

        Ranges for physical system vs. simulated system environments
        - radius: 150 -> 300 vs 0.2 -> 0.72
        - theta: 0 -> 180 vs -90 -> +90
        - z: 0 -> 150 vs 0.324 -> 1

        Parameters:
            obs (array): some observation parameter of the simulated system.

        Returns:
            obs: the observations in the physical system coordinates.
            
        """
        print("obs before: ", obs)
        obs[0]=(obs[0]-.2)*(RADIUS_LIMIT[MAX]-RADIUS_LIMIT[MIN])/(.72-.2) + RADIUS_LIMIT[MIN]
        obs[1]=(obs[1]+90)
        obs[2]=(obs[2]-.32)*(HEIGHT_LIMIT[MAX])/(1.0-.32)
        print("obs after: ", obs)
        return obs

    def get_obs_phys_to_sim(self, obs):
        """
        Convert positional observation parameters from the physical environment to the simulated system environment.

        Ranges for physical system vs. simulated system environments
        - radius: 150 -> 250 vs 0.2 -> 0.72
        - theta: 0 -> 180 vs -90 -> +90
        - z: 0 -> 150 vs 0.324 -> 1

        Parameters:
            obs (array): some observation parameter of the physical system.

        Returns:
            obs: the observations in the simulated system coordinates.
            
        """
        obs[0] = (obs[0] - RADIUS_LIMIT[MIN]) * (.72 - .2) / (RADIUS_LIMIT[MAX] - RADIUS_LIMIT[MIN]) + 0.2
        obs[1] = (obs[1] - 90)
        obs[2] = (obs[2]) * (1.0 - .32) / (HEIGHT_LIMIT[MAX]) + .32
        return obs

    def get_observation(self):
        """
        Determines the current observation space for the physical system.
        Currently unused.

        Returns:
            obs (array): the observations in the simulated system coordinates.
            
        """
        current_position = self.get_obs_sim_to_phys(np.array(self.uarm_controller.get_polar()))
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
        """
        Determines the current observation space for the simulated system.

        Returns:
            obs (array): the observations in the simulated system coordinates.
            
        """

        # Get the object position in the physical system and convert it to simulated coordinates
        obj_pos_polar = convertToPolar(self.object_pos[0], self.object_pos[1], self.object_pos[2])
        obj_pos_polar[1] += 90
        obj_pos_polar = self.get_obs_phys_to_sim(obj_pos_polar)

        # Current position of the Uarm in simulated environment coordinates
        current_position = self.get_obs_phys_to_sim(np.array(self.uarm_controller.get_polar()))

        # Convert all to cartesian for use in ddpg
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
        """
        Adds an offset for the X and Y positions in the simulated system environment.

        Returns:
            simulated_obs (array): the observations in the simulated system coordinates.
            
        """
        simulated_obs['observation'][0] += X_OFFSET
        simulated_obs['observation'][1] += Y_OFFSET

        simulated_obs['achieved_goal'][0] += X_OFFSET
        simulated_obs['achieved_goal'][1] += Y_OFFSET

        simulated_obs['desired_goal'][0] += X_OFFSET
        simulated_obs['desired_goal'][1] += Y_OFFSET

        return simulated_obs

    def get_object_pos(self):
        """
        Returns the object's position.    
        """
        return self.object_pos

    def set_object_pos(self, pos):
        """
        Sets the object's position.

        Parameters:
            pos (array): current position of the object   
        """
        self.object_pos = pos

    def goal_distance(self, achieved_goal, desired_goal):
        """
        Computes the distance between the desired goal and the achieved goal.

        Parameters:
            achieved_goal (array): the current position of the Uarm
            desired_goal (array): the position of the object and the desired_goal of the Uarm

        Returns:
            The norm of the vectors as a float representing the distance.
            
        """
        assert len(achieved_goal) == len(desired_goal)
        return np.linalg.norm(achieved_goal - desired_goal, axis=-1)

    def update_object_rel(self, uarm_pos):
        """
        Updates the relative position of the object.
        Currently unused.

        Parameters:
            uarm_pos (array): the current position of the Uarm

        Returns:
            The relative position of the object to the Uarm
            
        """
        # TODO make sure this subtraction is in the right order
        x = uarm_pos[0] - self.object_pos[0]
        y = uarm_pos[1] - self.object_pos[1]
        z = uarm_pos[2] - self.object_pos[2]
        self.object_rel_pos = np.array(self.uarm_controller.coordinate_to_angles(x, y, z))
