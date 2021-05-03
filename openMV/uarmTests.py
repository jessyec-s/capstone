import random
from uarmEnv import RADIUS_LIMIT, ANGLE_LIMIT, HEIGHT_LIMIT, MAX, MIN

class UarmTests():
    def __init__(self, **kwargs):
        super(UarmTests, self).__init__()

    def run_tests(self):
        # Add tests here in the future if necessary
        return self.generate_random_object_locations(50)

    def generate_random_object_locations(self, num_iters=50):
        print("inside determine_success_rate")
        locations = []
        for i in range(num_iters):
            locations.append([random.randrange(RADIUS_LIMIT[MIN], RADIUS_LIMIT[MAX]),
                             random.randrange(ANGLE_LIMIT[MIN], ANGLE_LIMIT[MAX]),
                             random.randrange(HEIGHT_LIMIT[MIN], HEIGHT_LIMIT[MAX])])
        return locations