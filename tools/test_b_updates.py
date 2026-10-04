"""B3 unit test: check the Q-learning updates of part_b/agent.py exactly on a
scripted fake environment (no simulator randomness involved).

Script of one episode (actions ignored by the fake env):
    s0 --a--> r=0.1 --> s1 --a--> r=0.2 --> sC (collided) --a--> r=-5, done
With alpha = 1 and greedy actions, after enough episodes:
  * Q[sC, :] stays exactly 0           (the collision step's action never ran)
  * Q[s1, a] = 0.2 + g * (-5)          (the -5 is credited to the action that caused it)
  * Q[s0, a] = 0.1 + g * max_a Q[s1]   (ordinary bootstrapped update)
A second script ends by the time limit instead (positive final reward), which the
agent credits the same way.
"""
import os, sys
PB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'A2-starter-code', 'A2-starter-code', 'part_b')
sys.path.insert(0, PB)
import numpy as np
import agent as agent_module
from env import HighwayEnv

S0 = (1, 2, [0, 0, 0, 0])
S1 = (1, 2, [0, 0, 2, 0])
SC = (1, 2, [0, 0, 1, 0])
SCRIPT = None


class FakeEnv:
    num_speed_states, num_lanes, num_dist_states, num_actions = 4, 4, 5, 5

    def __init__(self):
        self.t = 0

    def get_state(self):
        return SCRIPT[0][0]

    def step(self, action):
        s, r, done = SCRIPT[self.t + 1]
        self.t += 1
        return s, r, done


def run(script, g):
    global SCRIPT
    SCRIPT = script
    agent_module.HighwayEnv = FakeEnv
    ag = agent_module.Agent(HighwayEnv(), discount_factor=g)
    ag.alpha_start = ag.alpha_end = 1.0
    ag.eps_start = ag.eps_end = 0.0          # purely greedy
    ag.learn_policy(0.3)
    return ag


g = 0.9
# collision ending
ag = run([(S0, None, False), (S1, 0.1, False), (SC, 0.2, False), (SC, -5.0, True)], g)
i0, i1, iC = ag._index(*S0), ag._index(*S1), ag._index(*SC)
Q, visits = np.array(ag.Q), np.array(ag.visits)
tried1 = visits[i1] > 0
assert np.all(Q[iC] == 0.0), Q[iC]
assert np.all(visits[iC] == 0), visits[iC]
assert np.allclose(Q[i1][tried1], 0.2 + g * -5.0), Q[i1]
assert np.allclose(Q[i0][visits[i0] > 0], 0.1 + g * Q[i1].max()), (Q[i0], Q[i1])
print(f'collision episode: Q[s1] tried = {Q[i1][tried1]} (expected {0.2 + g * -5.0:.3f}), '
      f'Q[collided obs] = {Q[iC]} (expected all 0), episodes run = {ag.stats["episodes"]}')

# time-limit ending (positive final reward)
ag = run([(S0, None, False), (S1, 0.1, False), (SC, 0.2, False), (S0, 0.3, True)], g)
i0, i1, iC = ag._index(*S0), ag._index(*S1), ag._index(*SC)
Q, visits = np.array(ag.Q), np.array(ag.visits)
assert np.allclose(Q[i1][visits[i1] > 0], 0.2 + g * 0.3), Q[i1]
print(f'time-limit episode: Q[s1] tried = {Q[i1][visits[i1] > 0]} (expected {0.2 + g * 0.3:.3f})')
print('B3 update logic: OK')
