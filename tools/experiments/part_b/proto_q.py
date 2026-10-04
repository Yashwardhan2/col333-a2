"""Prototype Q-learning variants (scratch, for evaluating review claims only).
usage: python proto_q.py <variant> <T> <seed> <eval_episodes> [gamma]
variants:
  gemini        eps = max(0.01, 1 - t/(0.8T)), alpha = 0.1, Q0 = 0, standard terminal update
  gemini+fix    same, but the -5 returned at the collision step is credited to the action that
                caused it: Q(s_t,a_t) <- r_t + gamma*r_{t+1}; the collision step's own (s,a) is not updated
  fix+alpha_n   gemini+fix with alpha = 1/(1+n(s,a))^0.6
"""
import sys, time
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'A2-starter-code', 'A2-starter-code', 'part_b'))
from env import HighwayEnv

variant, T, seed, n_eval = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
g = float(sys.argv[5]) if len(sys.argv) > 5 else 0.99
rng = np.random.default_rng(seed); np.random.seed(seed)
CAP = 1000
fix = 'fix' in variant
alpha_n = variant.endswith('alpha_n')

Q = np.zeros((4, 4, 5, 5, 5, 5, 5)); N = np.zeros(Q.shape, dtype=np.int64)
def key(s): return (s[0], s[1], *s[2])
def upd(idx, a, target):
    sa = idx + (a,)
    N[sa] += 1
    al = (1.0 + N[sa]) ** -0.6 if alpha_n else 0.1
    Q[sa] += al * (target - Q[sa])

t0 = time.time(); deadline = t0 + 0.85 * T; steps = 0
while time.time() < deadline:
    env = HighwayEnv(); idx = key(env.get_state()); k = 0; pending = None
    while True:
        frac = (time.time() - t0) / (0.8 * T)
        eps = max(0.01, 1.0 - frac)
        a = int(rng.integers(5)) if rng.random() < eps else int(Q[idx].argmax())
        s2, r, done = env.step(a); k += 1; steps += 1
        idx2 = key(s2)
        collision = done and k < CAP
        if not fix:
            upd(idx, a, r if collision else r + g * Q[idx2].max())
        else:
            if collision:
                # this step's action could not matter; the -5 belongs to the previous action
                if pending is not None:
                    pi, pa, pr = pending
                    upd(pi, pa, pr + g * r)
                pending = None
            else:
                if pending is not None:
                    pi, pa, pr = pending
                    upd(pi, pa, pr + g * Q[idx].max())
                pending = (idx, a, r)
                if done:   # time-limit truncation: bootstrap the last transition
                    upd(idx, a, r + g * Q[idx2].max()); pending = None
        if done: break
        idx = idx2
learn = time.time() - t0

# greedy evaluation, mirrors run.py (fresh env per episode, discount df)
rets, crashes, lens = [], 0, []
for _ in range(n_eval):
    env = HighwayEnv(); s = env.get_state(); rs = []
    while not env.done:
        s, r, d = env.step(int(Q[key(s)].argmax())); rs.append(r)
    dr = 0.0
    for r in reversed(rs): dr = dr * g + r
    rets.append(dr); lens.append(len(rs)); crashes += rs[-1] < 0
rets = np.array(rets)
print(f'{variant:12s} T={T:.0f} seed={seed} gamma={g}: {steps/1e6:.2f}M steps in {learn:.0f}s | '
      f'score {rets.mean():.3f} +- {1.96*rets.std()/np.sqrt(n_eval):.3f} | crash rate {crashes/n_eval:.2f} | mean length {np.mean(lens):.0f}')
