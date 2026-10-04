"""Claim check: is the aggressive, crash-prone policy a real optimum of the discounted
objective? Trains like run.py (SIGALRM budget T), then for each greedy episode records
the discounted return split into driving reward and crash penalty, the undiscounted
return and the crash step. Uses only rewards/observations (a crash = an episode whose
last reward is negative), never simulator internals.

usage: python episode_stats.py <T> <episodes> [df] [label]   (AGENT_DIR / AGENT_SET as in eval_b.py)
"""
import os, sys, time, signal
import numpy as np
PB = os.environ.get('AGENT_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'A2-starter-code', 'A2-starter-code', 'part_b')
sys.path.insert(0, PB)
from env import HighwayEnv
from agent import Agent

T, episodes = int(sys.argv[1]), int(sys.argv[2])
df = float(sys.argv[3]) if len(sys.argv) > 3 else 0.99
label = sys.argv[4] if len(sys.argv) > 4 else ''


class TimeoutException(Exception):
    pass


def handler(*a):
    raise TimeoutException


agent = Agent(HighwayEnv(), discount_factor=df)
for kv in filter(None, os.environ.get('AGENT_SET', '').split(',')):
    k, v = kv.split('=')
    setattr(agent, k.strip(), type(getattr(agent, k.strip()))(v))
signal.signal(signal.SIGALRM, handler); signal.alarm(T)
try:
    agent.learn_policy(T)
except TimeoutException:
    print(f'{label} TIMEOUT'); sys.exit(1)
finally:
    signal.alarm(0)

disc, drive, crash_pen, undisc, crash_steps, speeds = [], [], [], [], [], []
for _ in range(episodes):
    env = HighwayEnv(); s = env.get_state(); rs = []; sp = []
    while not env.done:
        sp.append(s[0])
        s, r, d = env.step(agent.get_action(*s)); rs.append(r)
    g = df ** np.arange(len(rs))
    rs = np.array(rs)
    crashed = rs[-1] < 0
    disc.append(float((g * rs).sum()))
    crash_pen.append(float(g[-1] * rs[-1]) if crashed else 0.0)
    drive.append(disc[-1] - crash_pen[-1])
    undisc.append(float(rs.sum()))
    if crashed:
        crash_steps.append(len(rs))
    speeds.append(np.mean(sp))
disc = np.array(disc); n = episodes
q = np.percentile(crash_steps, [10, 25, 50, 75, 90]).astype(int) if crash_steps else []
print(f'{label} T={T} df={df}: discounted {disc.mean():.3f} +- {1.96*disc.std()/np.sqrt(n):.3f} '
      f'= driving {np.mean(drive):.3f} + crash penalty {np.mean(crash_pen):.3f} | undiscounted {np.mean(undisc):.2f} '
      f'+- {1.96*np.std(undisc)/np.sqrt(n):.2f} | crash rate {len(crash_steps)/n:.2f}, crash step p10/25/50/75/90 = '
      f'{list(q)} | mean speed idx {np.mean(speeds):.2f}')
