from uarm.wrapper import SwiftAPI
import numpy as np
import random
import math
import gym
from gym import spaces

MIN = 0
MAX = 1
RADIUS_LIMIT = (150., 250.)
ANGLE_LIMIT = (0., 180.)
HEIGHT_LIMIT = (0., 150.)

OBJECT_HEIGHT = 20  # mm
CAMERA_Z_OFFSET = 37
CAMERA_Y_OFFSET = 40


class UarmEnv(SwiftAPI, gym.GoalEnv):
    def __init__(self, port=None, baudrate=115200, timeout=None, **kwargs):
        super(UarmEnv, self).__init__()
        # super().__init__(port=None, baudrate=115200, timeout=None, **kwargs)
        self.toggle_dir = False
        self.distance_threshold = 0.05

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

    def init(self, port=None, baudrate=115200, timeout=10, **kwargs):
        super(self, port='/dev/cu.usbmodem142401', baudrate=115200, timeout=20, **kwargs)

    def UArm_reset(self, should_wait=False):
        radius = random.uniform(RADIUS_LIMIT[MIN], RADIUS_LIMIT[MAX])
        angle = random.uniform(ANGLE_LIMIT[MIN], ANGLE_LIMIT[MAX])
        height = random.uniform(HEIGHT_LIMIT[MIN], HEIGHT_LIMIT[MAX])
        print("ENV_RESET: radius: ", radius, "angle: ", angle, "height: ", height)
        return self.set_polar(radius, angle, height, wait=should_wait)

    def reset(self):
        self.UArm_reset(True)
        self.do_suction(False)
        # self.object_pos = np.zeros(3)  # in cartesian coordinates
        self.update_object_rel(self.get_position())
        self.suction_state = np.zeros(1)
        print("RESET RETURNS: ", self.get_observation())
        return self.get_observation()  # reward, done, info can't be included

    def calc_distance_to_object(self, cam_h_angle, cam_v_angle):
        coord = self.get_position()
        print("CORD: ", coord)
        if coord is None:
            print("Error getting robot's position")
            return
        rel_z = coord[2] - OBJECT_HEIGHT + CAMERA_Z_OFFSET
        distance = math.sqrt(rel_z ** 2 + (rel_z * math.tan(cam_v_angle * math.pi / 180)) ** 2
                             + (rel_z * math.tan(cam_h_angle * math.pi / 180)) ** 2)
        print("distance to object: ", distance)
        return distance

    def calc_object_cords(self, cam_h_angle, cam_v_angle):
        coord = self.get_position()
        if coord is None:
            print("Error getting robot's position")
            return
        rel_z = coord[2] - OBJECT_HEIGHT + CAMERA_Z_OFFSET
        y = rel_z * math.tan(cam_v_angle * math.pi / 180) + coord[1] + CAMERA_Y_OFFSET
        x = rel_z * math.tan(cam_h_angle * math.pi / 180) + coord[0]

        # TODO: maybe convert to polar
        self.object_pos = np.array([x, y, OBJECT_HEIGHT])
        print("Setting object position to: ", [x, y, OBJECT_HEIGHT])

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        return -(d > self.distance_threshold).astype(np.float32)

    def do_suction(self, do_suction):
        self.set_suction_state(True)
        self.set_pump(on=do_suction, wait=True)

    def is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def seek(self):
        # toggle controls whether we want to switch directions or not
        print("in seek")
        curr_pos = self.get_polar()
        if curr_pos is None or curr_pos == "TIMEOUT":
            print("Seek returning: ", curr_pos)
            return False
        print("Cur pos: ", curr_pos)

        radius = curr_pos[0]  # stretch in the docs
        angle = curr_pos[1]  # rotation in the docs
        height = curr_pos[2]

        print(radius, angle, height)
        if height < 30:
            height = 100
        if self.toggle_dir:
            angle -= 10
        else:
            angle += 10

        # angle boundaries- may need to extend to the full range (0-180)
        if angle > 150:
            angle = 150
            self.toggle_dir = True
        elif angle < 30:
            angle = 30
            self.toggle_dir = False
        self.set_polar(radius, angle, height, wait=True)
        print("--- Setting new position ---")
        print("radius: ", radius, "angle: ", angle, "height: ", height)

    def step(self, u, is_polar=False):
        lastPos = self.get_polar()
        if not is_polar:
            u = self.coordinate_to_angles(u)
        # clip at maximum positions
        print("U: ", u)
        new_u = [sum(x) for x in zip(lastPos, u)]
        if not self.check_pos_is_limit(new_u, is_polar=True):
            new_u[0] = np.clip(new_u[0], RADIUS_LIMIT[MIN], RADIUS_LIMIT[MAX])
            new_u[1] = np.clip(new_u[1], ANGLE_LIMIT[MIN], ANGLE_LIMIT[MAX])
            new_u[2] = np.clip(new_u[2], HEIGHT_LIMIT[MIN], HEIGHT_LIMIT[MAX])

        self.set_polar(stretch=new_u[0], rotation=new_u[1], height=new_u[2], wait=True)
        self.update_object_rel(self.get_position())

        desired_goal = self.object_pos
        achieved_goal = self.get_polar()

        return self.get_observation(), self.compute_reward(achieved_goal, desired_goal), self.is_success(achieved_goal, desired_goal)

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
            'achieved_goal': np.array(self.get_polar()),
            'desired_goal': np.array(self.object_pos),
        }

    def get_object_pos(self):
        return self.object_pos

    def set_object_pos(self, pos):
        # maybe convert to polar
        self.object_pos = pos

    def update_object_rel(self, uarm_pos):
        # TODO make sure this subtraction is in the right order
        x = uarm_pos[0] - self.object_pos[0]
        y = uarm_pos[1] - self.object_pos[1]
        z = uarm_pos[2] - self.object_pos[2]
        self.object_rel_pos = np.array([x, y, z])

    def set_suction_state(self, state):
        self.suction_state = state

    def goal_distance(self, goal_a, goal_b):
        assert len(goal_a) == len(goal_b)
        return np.linalg.norm(goal_a - goal_b, axis=-1)

