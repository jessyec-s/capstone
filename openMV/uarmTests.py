from threaded import ddpg_loop_no_camera
import random

RADIUS_LIMIT = (150., 250.)
ANGLE_LIMIT = (0., 180.)
HEIGHT_LIMIT = (0., 150.)


def run_tests():
    determine_success_rate()


def determine_success_rate(num_iters=50):
    locations = []
    for i in range(num_iters):
        locations.append(random.randrange(RADIUS_LIMIT[0], RADIUS_LIMIT[1]),
                         random.randrange(ANGLE_LIMIT[0], ANGLE_LIMIT[1]),
                         random.randrange(HEIGHT_LIMIT[0], HEIGHT_LIMIT[1]))
    ddpg_loop_no_camera(locations)


run_tests()
