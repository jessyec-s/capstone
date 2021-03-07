#just a test function for reward calculation
eo=.15
alpha=.8
s=500
em=.05
def calc_e(k):
    return em +(eo-em)*((s-k)/s)**alpha

def goal_distance(goal_a, goal_b):
    assert len(goal_a) == len(goal_b)
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def reward_calc(goal_a,goal_b,k):
    goal=goal_distance(goal_a,goal_b)
    if goal>calc_e(k):
        return -goal
    else:
        return 1

