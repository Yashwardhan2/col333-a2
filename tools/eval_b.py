"""Part B evaluation harness: run.py-equivalent (SIGALRM budget T, fresh HighwayEnv
per episode, discount df) without GIF rendering, with more episodes and diagnostics.

usage: python tools/eval_b.py <T> <episodes> [df] [label]
Diagnostics read only rewards/observations (crash = an episode whose last reward is
negative; used for reporting only, never by the agent).
"""
import os, sys, time, signal
import numpy as np
PB = os.environ.get('AGENT_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'A2-starter-code', 'A2-starter-code', 'part_b')
sys.path.insert(0, PB)
from env import HighwayEnv
from agent import Agent

T, episodes = int(sys.argv[1]), int(sys.argv[2])
df = float(sys.argv[3]) if len(sys.argv) > 3 else 0.99
label = sys.argv[4] if len(sys.argv) > 4 else ''


class TimeoutException(Exception):   # same name as run.py's
    pass


def handler(*a):
    raise TimeoutException


env = HighwayEnv()
agent = Agent(env, discount_factor=df)
signal.signal(signal.SIGALRM, handler); signal.alarm(T)
t0 = time.time()
try:
    agent.learn_policy(T)
except TimeoutException:
    print(f'{label} TIMEOUT -> run.py would score None'); sys.exit(1)
finally:
    signal.alarm(0)
learn = time.time() - t0
steps = getattr(agent, 'stats', {}).get('steps', 0)

rets, lens, crashes, speeds, unvisited, total = [], [], 0, [], 0, 0
visits = getattr(agent, 'visits', None)
for _ in range(episodes):
    env = HighwayEnv(); s = env.get_state(); rs = []
    while not env.done:
        if visits is not None:
            unvisited += int(visits[agent._index(*s)].sum() == 0)
        total += 1; speeds.append(s[0])
        s, r, d = env.step(agent.get_action(*s)); rs.append(r)
    dr = 0.0
    for r in reversed(rs): dr = dr * df + r
    rets.append(dr); lens.append(len(rs)); crashes += rs[-1] < 0
rets = np.array(rets)
print(f'{label} T={T} df={df}: learn {learn:.1f}s, {steps/1e6:.2f}M steps ({steps/max(learn,1e-9)/1e3:.1f}k/s) | '
      f'score {rets.mean():.3f} +- {1.96*rets.std()/np.sqrt(episodes):.3f} ({episodes} eps) | crash rate {crashes/episodes:.2f} | '
      f'mean length {np.mean(lens):.0f} | mean speed idx {np.mean(speeds):.2f} | unvisited-state steps {100*unvisited/max(total,1):.2f}%')
