"""run.py-equivalent harness without GIF rendering: same SIGALRM budget, same
2N^2 cap and discounting, but many more episodes."""
import sys, os, time, signal
import numpy as np
PA = os.environ.get('AGENT_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'A2-starter-code', 'A2-starter-code', 'part_a')
sys.path.insert(0, PA)
from env import TreasureHunt
from agent import Agent
lay, pr, T, runs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lay, pr = os.path.abspath(lay), os.path.abspath(pr)
os.chdir(PA)
class TimeoutException(Exception): pass   # same name as run.py's
def h(*a): raise TimeoutException
t0 = time.time(); ag = Agent(lay, pr); tinit = time.time() - t0
signal.signal(signal.SIGALRM, h); signal.alarm(T)
t0 = time.time()
try:
    ag.learn_policy(T)
except TimeoutException:
    print('TIMEOUT -> score None'); sys.exit(1)
finally:
    signal.alarm(0)
tl = time.time() - t0
rets = []
for _ in range(runs):
    env = TreasureHunt(lay, pr); s = env.get_state(); rs = []; k = 0
    while not env.done and k < 2 * env.N ** 2:
        s, r, d = env.step(ag.get_action(*s)); rs.append(r); k += 1
    g = 0.0
    for r in reversed(rs): g = g * env.df + r
    rets.append(g)
rets = np.array(rets)
print(f'{os.path.basename(os.path.dirname(lay)) or lay}: init {tinit:.2f}s learn {tl:.1f}s/{T}s  '
      f'V(start)={ag.V[ag._encode(*TreasureHunt(lay, pr).get_state())]:.4f}  '
      f'score={rets.mean():.4f} +- {1.96*rets.std()/np.sqrt(runs):.4f} ({runs} runs)')
