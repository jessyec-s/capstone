# from threaded import ddpg_loop_no_camera
import random

RADIUS_LIMIT = (150., 250.)
ANGLE_LIMIT = (0., 180.)
HEIGHT_LIMIT = (0., 150.)

class UarmTests():
    def __init__(self, **kwargs):
        super(UarmTests, self).__init__()

    def run_tests(self):
        # Add tests here in the future if necessary
        return self.generate_random_object_locations(10)

    def generate_random_object_locations(self, num_iters=50):
        print("inside determine_success_rate")
        locations = []
        for i in range(num_iters):
            locations.append([random.randrange(RADIUS_LIMIT[0], RADIUS_LIMIT[1]),
                             random.randrange(ANGLE_LIMIT[0], ANGLE_LIMIT[1]),
                             random.randrange(HEIGHT_LIMIT[0], HEIGHT_LIMIT[1])])
        return locations