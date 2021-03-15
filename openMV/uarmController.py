from uarm.wrapper import SwiftAPI
import random
import math
import numpy as np


MIN = 0
MAX = 1
RADIUS_LIMIT = (150., 250.)
ANGLE_LIMIT = (0., 180.)
HEIGHT_LIMIT = (0., 150.)

OBJECT_HEIGHT = 20  # mm
CAMERA_Z_OFFSET = 37
CAMERA_Y_OFFSET = 40

class UarmController(SwiftAPI):
    def __init__(self, port=None, baudrate=115200, timeout=None, **kwargs):
        super(UarmController, self).__init__()
        self.toggle_dir = False

    def init(self, port=None, baudrate=115200, timeout=10, **kwargs):
        super(self, port='/dev/cu.usbmodem142401', baudrate=115200, timeout=20, **kwargs)

    def UArm_reset(self, should_wait=False):
        # [216.80258237229708, 89.9080001961666, 46.718538609349665]
        # radius = random.uniform(RADIUS_LIMIT[MIN], RADIUS_LIMIT[MAX])
        # angle = random.uniform(ANGLE_LIMIT[MIN], ANGLE_LIMIT[MAX])
        # height = random.uniform(HEIGHT_LIMIT[MIN], HEIGHT_LIMIT[MAX])
        # print("ENV_RESET: radius: ", radius, "angle: ", angle, "height: ", height)
        return self.set_polar(216.80258237229708, 89.9080001961666, 46.718538609349665, wait=should_wait)

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
        print("Setting object position to: ", [x, y, OBJECT_HEIGHT])
        return np.array([x, y, OBJECT_HEIGHT])

    def do_suction(self, do_suction):
        self.set_pump(on=do_suction, wait=True)

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
