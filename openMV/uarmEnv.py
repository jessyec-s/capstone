import numpy as np
import gym
from gym import spaces

MIN = 0
MAX = 1
RADIUS_LIMIT = (150., 250.)
ANGLE_LIMIT = (0., 180.)
HEIGHT_LIMIT = (0., 150.)

class UarmEnv(gym.GoalEnv):
    def __init__(self, uarm_controller, timeout=None, **kwargs):
        super(UarmEnv, self).__init__()

        self.distance_threshold = 0.05
        self.uarm_controller = uarm_controller

        self.object_pos = np.zeros(3)  # in cartesian coordinates
        self.object_rel_pos = np.zeros(3)
        self.suction_state = np.zeros(1)
        self.suction_pos = np.zeros(3)

        # Define action and observation space
        # TODO: not sure if the shape here is the right value
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype='float32')

        obs = self.get_observation()

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def reset(self):
        self.uarm_controller.UArm_reset(True)
        self.uarm_controller.do_suction(False)
        self.suction_state = False
        # self.object_pos = np.zeros(3)  # in cartesian coordinates
        self.update_object_rel(self.uarm_controller.get_position())
        self.suction_state = np.zeros(1)
        print("RESET RETURNS: ", self.get_observation())
        return self.get_observation()  # reward, done, info can't be included

    def compute_reward(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        return -(d > self.distance_threshold).astype(np.float32)

    def is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def step(self, u, is_polar=False):
        lastPos = self.uarm_controller.get_polar()
        # lastPos = [1, 1, 1]
        if not is_polar:
            u = self.uarm_controller.coordinate_to_angles(u)
        # clip at maximum positions
        print("U: ", u)
        new_u = [sum(x) for x in zip(lastPos, u)]
        if not self.uarm_controller.check_pos_is_limit(new_u, is_polar=True):
            new_u[0] = np.clip(new_u[0], RADIUS_LIMIT[MIN], RADIUS_LIMIT[MAX])
            new_u[1] = np.clip(new_u[1], ANGLE_LIMIT[MIN], ANGLE_LIMIT[MAX])
            new_u[2] = np.clip(new_u[2], HEIGHT_LIMIT[MIN], HEIGHT_LIMIT[MAX])

        self.uarm_controller.set_polar(stretch=new_u[0], rotation=new_u[1], height=new_u[2], wait=True)
        self.update_object_rel(self.uarm_controller.get_position())

        desired_goal = self.object_pos
        achieved_goal = self.uarm_controller.get_polar()

        return self.get_observation(), self.compute_reward(achieved_goal, desired_goal), self.is_success(achieved_goal, desired_goal), {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_observation(self):
        obs = np.concatenate([
            self.object_pos, self.object_rel_pos, self.suction_state, self.suction_pos,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': np.array(self.uarm_controller.get_polar()),
            'desired_goal': np.array(self.object_pos),
        }

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
        self.object_rel_pos = np.array([x, y, z])

