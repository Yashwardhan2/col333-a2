"""Diagnostics for the Part B review claims. Reads simulator internals for MEASUREMENT ONLY."""
import sys, time
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'A2-starter-code', 'A2-starter-code', 'part_b'))
from env import HighwayEnv, COLLISION_THRESH_CONTROL_CAR

rng = np.random.default_rng(0)

# ---- D1: speed of a lean epsilon-greedy Q-learning loop (fresh env per episode) ----
Q = np.zeros((4, 4, 5, 5, 5, 5, 5)); g = 0.99; alpha = 0.1; eps = 0.3
env = HighwayEnv(); s = env.get_state(); k = 0; n = 0; t0 = time.time()
while time.time() - t0 < 20:
    idx = (s[0], s[1], *s[2])
    a = int(rng.integers(5)) if rng.random() < eps else int(Q[idx].argmax())
    s2, r, done = env.step(a); k += 1; n += 1
    idx2 = (s2[0], s2[1], *s2[2])
    term = done and k < 1000
    target = r if term else r + g * Q[idx2].max()
    Q[idx + (a,)] += alpha * (target - Q[idx + (a,)])
    if done: env = HighwayEnv(); s = env.get_state(); k = 0
    else: s = s2
dt = time.time() - t0
print(f'D1 lean Q-learning loop: {n/dt:,.0f} steps/s  -> {0.85*60*n/dt/1e6:.2f}M steps in 0.85*60 s, {0.85*240*n/dt/1e6:.2f}M in 0.85*240 s')

# ---- D2/D3: coverage and aliasing under random and under the learned-ish policy above ----
def rollout(policy, steps):
    visits = {}; own1 = {'collided': 0, 'ahead_close': 0}; coll_obs = []; obs_count = {}
    env = HighwayEnv(); s = env.get_state(); k = 0; collisions = 0; episodes = 0
    for _ in range(steps):
        a = policy(s)
        s2, r, done = env.step(a); k += 1
        key = (s2[0], s2[1], *s2[2])
        if done and k < 1000:
            collisions += 1; coll_obs.append(key)
        elif not done:
            visits[key] = visits.get(key, 0) + 1
            # ground truth of the configuration behind this observation
            collided = env.check_collision(env.control_car, env.control_car.lane_id, COLLISION_THRESH_CONTROL_CAR)
            if s2[2][s2[1]] == 1:
                own1['collided' if collided else 'ahead_close'] += 1
            obs_count.setdefault(key, [0, 0])[1 if collided else 0] += 1
        if done: env = HighwayEnv(); s = env.get_state(); k = 0; episodes += 1
        else: s = s2
    return visits, own1, coll_obs, obs_count, collisions, episodes

for name, pol in [('random', lambda s: int(rng.integers(5))),
                  ('eps=0.1 greedy on Q above', lambda s: int(rng.integers(5)) if rng.random() < 0.1 else int(Q[(s[0], s[1], *s[2])].argmax()))]:
    visits, own1, coll_obs, obs_count, collisions, episodes = rollout(pol, 300000)
    v = np.array(sorted(visits.values(), reverse=True)); tot = v.sum()
    top = lambda f: v[:max(1, int(len(v) * f))].sum() / tot
    print(f'\nD2 [{name}] 300k steps, {episodes} episodes, {collisions} collisions: '
          f'{len(v)} distinct non-terminal observations of 10,000 '
          f'({(v >= 36*5).sum()} seen >= 180 times); top 10% of states hold {100*top(0.1):.0f}% of visits')
    n1 = own1['collided'] + own1['ahead_close']
    print(f'D3 [{name}] own-lane min_dist==1 (non-terminal steps): {n1} obs; '
          f'{100*own1["collided"]/max(n1,1):.1f}% are ALREADY-COLLIDED configurations (next step is -5 whatever the action), '
          f'{100*own1["ahead_close"]/max(n1,1):.1f}% are a car 0.5-1.67 ahead (action still matters)')
    own_at_coll = [o[2 + o[1]] for o in coll_obs]
    print(f'D3 [{name}] own-lane min_dist in the observation returned with the -5: {dict(zip(*np.unique(own_at_coll, return_counts=True)))}')
    mixed = [c for c in obs_count.values() if c[1] > 0]
    frac_mixed = sum(c[0] for c in mixed) / max(1, sum(c[0] + c[1] for c in mixed))
    print(f'D3 [{name}] observations that are sometimes collided: {len(mixed)}; within them '
          f'{100*frac_mixed:.0f}% of visits are NOT collided (so their Q cannot simply become -5)')
