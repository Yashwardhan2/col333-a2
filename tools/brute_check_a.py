"""Independent reference: enumerate the MDP explicitly from env.py's own methods,
run plain (unfactored) value iteration, and compare with the agent's V."""
import sys, os, itertools, time
import numpy as np
PA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'A2-starter-code', 'A2-starter-code', 'part_a')
sys.path.insert(0, PA); os.chdir(PA)
from env import TreasureHunt
from agent import Agent

def build(lay, pr):
    env = TreasureHunt(lay, pr)
    N = env.N
    ship_cells = [(i, j) for i in range(N) for j in range(N) if env.ship_loc_validity((i, j))]
    R1, R2 = env.pirate_areas
    T0 = list(env.original_locations['treasure'])
    tsets = [tuple(t for i, t in enumerate(T0) if keep[i]) for keep in itertools.product([0, 1], repeat=len(T0))]
    states = [(s, p1, p2, ts) for ts in tsets for s in ship_cells for p1 in R1 for p2 in R2]
    sid = {x: k for k, x in enumerate(states)}
    ps = env.ship_prob[0]

    def pirate_dist(loc, probs):
        out = {}
        for a in range(4):
            if probs[a] == 0: continue
            n = env.move(loc, a)
            if n not in env.locations['pirate_area']: n = loc
            out[n] = out.get(n, 0) + probs[a]
        return out

    rows, cols, probs, rews, acts = [], [], [], [], []
    for k, (s, p1, p2, ts) in enumerate(states):
        d1 = pirate_dist(p1, env.pirate_prob[0]); d2 = pirate_dist(p2, env.pirate_prob[1])
        for a in range(4):
            for e in range(4):
                pe = ps if e == a else (1 - ps) / 3
                n = env.move(s, e)
                if not env.ship_loc_validity(n): n = s
                for n1, q1 in d1.items():
                    for n2, q2 in d2.items():
                        p = pe * q1 * q2
                        r = env.rewards['step']; nxt = -1; nts = ts
                        if n in (n1, n2): r += env.rewards['pirate']
                        elif n in env.locations['fort']: r += env.rewards['fort']
                        else:
                            if n in ts:
                                r += env.rewards['treasure']; nts = tuple(t for t in ts if t != n)
                            nxt = sid[(n, n1, n2, nts)]
                        rows.append(k * 4 + a); cols.append(nxt); probs.append(p); rews.append(r)
    return env, states, sid, np.array(rows), np.array(cols), np.array(probs), np.array(rews)

def solve(env, states, rows, cols, probs, rews, tol=1e-11):
    g = env.df; S = len(states)
    V = np.zeros(S)
    term = cols < 0
    c = np.where(term, 0, cols)
    for it in range(100000):
        contrib = probs * (rews + g * np.where(term, 0.0, V[c]))
        Q = np.bincount(rows, weights=contrib, minlength=S * 4).reshape(S, 4)
        Vn = Q.max(1)
        d = np.abs(Vn - V).max(); V = Vn
        if d < tol: break
    return V, Q

def check(lay, pr):
    env, states, sid, rows, cols, probs, rews = build(lay, pr)
    V, Q = solve(env, states, rows, cols, probs, rews)
    ag = Agent(lay, pr); ag.learn_policy(600)
    worst = 0.0; bad = 0
    for k, (s, p1, p2, ts) in enumerate(states):
        if s in (p1, p2) or s in env.locations['fort'] or s in ts: continue
        idx = ag._encode(s, [p1, p2], list(ts))
        worst = max(worst, abs(ag.V[idx] - V[k]))
        if Q[k, ag.policy[idx]] < Q[k].max() - 1e-7: bad += 1
    return worst, bad, len(states)

if __name__ == '__main__':
    for test in sys.argv[1:]:
        print(test, check(f'tests/{test}/layout.txt', f'tests/{test}/prob.txt'))
