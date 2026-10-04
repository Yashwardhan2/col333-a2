"""Claim check: is interrupting learn_policy with run.py's SIGALRM harmless once the
agent catches TimeoutException? Fires the alarm at random moments (anywhere in the
training loop, including inside env.step and in the middle of a Q update) and checks
that learn_policy returns normally and the Q-table/visit counts stay valid.

usage: AGENT_DIR=<dir with an agent that has the safety net> python timeout_stress.py <trials>
"""
import os, sys, math, random, signal, time
PB = os.environ['AGENT_DIR']
sys.path.insert(0, PB)
from env import HighwayEnv
from agent import Agent


class TimeoutException(Exception):     # same class name as run.py's
    pass


def handler(*a):
    raise TimeoutException


trials = int(sys.argv[1])
rng = random.Random(1)
signal.signal(signal.SIGALRM, handler)
bad, overruns, escaped = 0, [], 0
for k in range(trials):
    ag = Agent(HighwayEnv(), discount_factor=rng.choice([0.5, 0.9, 0.99, 0.999]))
    ag.guard_frac = 10.0                      # never stop by itself: the alarm must end training
    for round_ in range(2):                   # interrupt, then resume training and interrupt again
        when = rng.uniform(0.02, 0.4)
        signal.setitimer(signal.ITIMER_REAL, when)
        t0 = time.time()
        try:
            ag.learn_policy(1.0)
        except TimeoutException:
            escaped += 1
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        overruns.append(time.time() - t0 - when)
    ok = all(isinstance(q, float) and math.isfinite(q) for row in ag.Q for q in row)
    ok &= all(isinstance(v, int) and v >= 0 for row in ag.visits for v in row)
    for _ in range(50):
        a = ag.get_action(rng.randrange(4), rng.randrange(4), [rng.randrange(5) for _ in range(4)])
        ok &= isinstance(a, int) and 0 <= a < 5
    bad += not ok
overruns.sort()
print(f'{trials} agents x 2 interruptions: exceptions escaping learn_policy={escaped}, '
      f'invalid Q/visits/actions={bad}, return delay after the alarm: median {1000*overruns[len(overruns)//2]:.2f} ms, '
      f'max {1000*overruns[-1]:.2f} ms')
