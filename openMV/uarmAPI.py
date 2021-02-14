from uarm.wrapper import SwiftAPI
import numpy as np
import random

MIN=0
MAX=1
RADIUS_LIMIT=(150.,250.)
ANGLE_LIMIT=(0.,180.)
HEIGHT_LIMIT=(0.,150.)


def goal_distance(goal_a, goal_b):
    assert len(goal_a) == len(goal_b)
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class UarmEnv(SwiftAPI):
    def __init__(self, port=None, baudrate=115200, timeout=None, **kwargs):
        super().__init__(port=None, baudrate=115200, timeout=None, **kwargs)
        self.toggle_dir = False

    def init(self, port=None, baudrate=115200, timeout=10, **kwargs):
        super(self, port='/dev/cu.usbmodem142401', baudrate=115200, timeout=20, **kwargs)

    def ENV_reset(self, should_wait=False):
        radius=random.uniform(RADIUS_LIMIT[MIN],RADIUS_LIMIT[MAX])
        angle=random.uniform(ANGLE_LIMIT[MIN],ANGLE_LIMIT[MAX])
        height=random.uniform(HEIGHT_LIMIT[MIN],HEIGHT_LIMIT[MAX])
        print("ENV_RESET: radius: ",radius, "angle: ",angle,"height: ",height)
        return self.set_polar(radius,angle,height,wait=should_wait)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        return -(d > self.distance_threshold).astype(np.float32)


    def get_camera_data(self):
        pass

    def is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
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
        returned_value = self.set_polar(radius, angle, height, wait=True)
        print("--- Setting new position ---")
        print("radius: ", radius, "angle: ", angle, "height: ", height)

    def step(self, u,is_polar=False):
        lastPos=self.get_polar()
        if not is_polar:
            u = self.coordinate_to_angles(u)
        # clip at maximum positions
        new_u=[sum(x) for x in zip(lastPos,u)]
        if not self.check_pos_is_limit(new_u,is_polar=True):
            new_u[0]=np.clip(new_u[0],RADIUS_LIMIT[MIN],RADIUS_LIMIT[MAX])
            new_u[1]=np.clip(new_u[1], ANGLE_LIMIT[MIN], ANGLE_LIMIT[MAX])
            new_u[2]=np.clip(new_u[2], HEIGHT_LIMIT[MIN], HEIGHT_LIMIT[MAX])

        # done = self.is_success(new_u,[0,0,0])
        # cost = self.compute_reward(new_u,[0,0,0],)
        self.set_polar(stretch=new_u[0],rotation=new_u[1],height=new_u[2])

        return {"observation": self.get_polar(),"desiredGoal":self.get_camera_data(),"AchievedGoal":self.get_polar()}




