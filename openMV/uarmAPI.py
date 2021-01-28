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
    def init(self, port=None, baudrate=115200, timeout=None, **kwargs):
        super(self, port=None, baudrate=115200, timeout=None, **kwargs)

    def ENV_reset(self):
        radius=random.uniform(RADIUS_LIMIT[MIN],RADIUS_LIMIT[MAX])
        angle=random.uniform(ANGLE_LIMIT[MIN],ANGLE_LIMIT[MAX])
        height=random.uniform(HEIGHT_LIMIT[MIN],HEIGHT_LIMIT[MAX])
        print("radius: ",radius, "angle: ",angle,"height: ",height)
        self.set_polar(radius,angle,height)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        return -(d > self.distance_threshold).astype(np.float32)


    #
    def get_camera_data(self):
        pass

    def is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

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

        return {"observation":self.get_polar(),"desiredGoal":self.get_camera_data(),"AchievedGoal":self.get_polar()}




