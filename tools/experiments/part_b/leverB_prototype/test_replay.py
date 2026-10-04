import random, sys, copy
sys.path.insert(0, '.')
from env import HighwayEnv
from agent import Agent

def online_reference(Q, episodes, g, alpha):
    # mirrors agent._train's pending/delayed-credit logic, written independently
    for I, A, R in episodes:
        pending = None
        for k in range(len(I)):
            i, a, r = I[k], A[k], R[k]
            done = (k == len(I) - 1)
            row = Q[i]
            if done:
                if pending is not None:
                    pi, pa, pr = pending
                    Q[pi][pa] += alpha * (pr + g * r - Q[pi][pa])
                break
            if pending is not None:
                pi, pa, pr = pending
                Q[pi][pa] += alpha * (pr + g * max(row) - Q[pi][pa])
            pending = (i, a, r)
    return Q

rng = random.Random(3)
worst = 0.0
for trial in range(300):
    ag = Agent(HighwayEnv(), discount_factor=rng.choice([0.5, 0.9, 0.99, 0.999]))
    U = rng.choice([0.0, 0.5, 3.0])
    for row in ag.Q: row[:] = [U] * ag.n_actions
    states = [rng.randrange(ag.n_states) for _ in range(6)]       # few states -> lots of revisits
    eps = []
    for _ in range(rng.randint(1, 4)):
        n = rng.randint(1, 8)
        eps.append(([rng.choice(states) for _ in range(n)], [rng.randrange(5) for _ in range(n)],
                    [rng.choice([0.03, 0.06, 0.09, 0.12]) for _ in range(n - 1)] + [rng.choice([-5.0, 0.12])]))
    ref = online_reference(copy.deepcopy(ag.Q), eps, ag.gamma, 0.37)
    ag._replay(eps, 0.37)
    worst = max(worst, max(abs(x - y) for s in states for x, y in zip(ag.Q[s], ref[s])))
print(f'replay vs independent online reference over 300 random cases: max |diff| = {worst:.3e}')
assert worst < 1e-12
print('replay equivalence: OK')
