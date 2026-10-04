"""A7 stress test: random layouts (N up to 30, small and large pirate regions,
random wind / pirate probabilities / rewards / gamma) run like run.py
(SIGALRM budget T), then checked against Monte-Carlo rollouts in env.py.

usage: python stress_a.py <case seed> <T> <episodes> [N]
prints one line per case.
"""
import os, sys, time, signal, random, resource, tempfile
from collections import deque
import numpy as np
PA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'A2-starter-code', 'A2-starter-code', 'part_a')
sys.path.insert(0, PA)
from env import TreasureHunt
from agent import Agent

D = [(1, 0), (-1, 0), (0, -1), (0, 1)]


def blob(rng, N, size, banned):
    cells = [(i, j) for i in range(N) for j in range(N) if (i, j) not in banned]
    start = rng.choice(cells); reg = [start]; seen = {start}; tries = 0
    while len(reg) < size and tries < 50 * size:
        tries += 1
        i, j = rng.choice(reg); di, dj = rng.choice(D); n = (i + di, j + dj)
        if 0 <= n[0] < N and 0 <= n[1] < N and n not in seen and n not in banned:
            reg.append(n); seen.add(n)
    return reg


def generate(seed, N):
    rng = random.Random(seed)
    while True:
        big = rng.random() < 0.5
        s1 = rng.randint(10, 60) if big else rng.randint(1, 6)
        s2 = rng.randint(10, 60) if big else rng.randint(1, 6)
        r1 = blob(rng, N, s1, set())
        moat = set(r1) | {(i + di, j + dj) for (i, j) in r1 for di, dj in D}
        r2 = blob(rng, N, s2, moat)
        if any((i + di, j + dj) in set(r1) for (i, j) in r2 for di, dj in D):
            continue
        g = {(i, j): 'W' for i in range(N) for j in range(N)}
        for c in r1 + r2: g[c] = '!'
        g[r1[0]] = '1'; g[r2[0]] = '2'
        land_p = rng.uniform(0.0, 0.3)
        free = [c for c in g if g[c] == 'W']
        for c in free:
            if rng.random() < land_p: g[c] = 'L'
        water = [c for c in g if g[c] == 'W']
        if len(water) < 4: continue
        S, F, T1, T2 = rng.sample(water, 4)
        g[S], g[F], g[T1], g[T2] = 'S', 'F', 'T', 'T'
        # fort and both treasures must be reachable from the ship (pirate cells are passable)
        seen = {S}; dq = deque([S])
        while dq:
            i, j = dq.popleft()
            for di, dj in D:
                n = (i + di, j + dj)
                if n in g and g[n] != 'L' and n not in seen: seen.add(n); dq.append(n)
        if not {F, T1, T2} <= seen: continue
        def pp():
            v = [rng.choice([0, 1, 1, 2, 3]) for _ in range(4)]
            if sum(v) == 0: v[rng.randrange(4)] = 1
            return ' '.join(f'{x / sum(v):.6f}' for x in v)
        rs = rng.choice([-0.01, -0.05, -0.1]); rt = rng.choice([1, 3, 5])
        rf = rng.choice([2, 5, 10]); rp = rng.choice([-1, -5, -10])
        gamma = rng.choice([0.9, 0.95, 0.99])
        ps = round(rng.uniform(0.5, 0.95), 3)
        d = tempfile.mkdtemp(prefix=f'stress_{seed}_')
        lay, pr = os.path.join(d, 'layout.txt'), os.path.join(d, 'prob.txt')
        with open(lay, 'w') as f:
            f.write('\n'.join(''.join(g[(i, j)] for j in range(N)) for i in range(N)) + '\n')
        with open(pr, 'w') as f:
            f.write(f'{ps}\n{pp()}\n{pp()}\n{rs} {rt} {rf} {rp}\n{gamma}\n')
        return lay, pr, (len(r1), len(r2), ps, gamma)


class TimeoutException(Exception):   # same name as run.py's
    pass


def handler(*a):
    raise TimeoutException


def main():
    seed, T, episodes = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    N = int(sys.argv[4]) if len(sys.argv) > 4 else random.Random(seed).choice([10, 20, 30])
    lay, pr, (s1, s2, ps, gamma) = generate(seed, N)
    env = TreasureHunt(lay, pr)            # env.py's own asserts must accept the layout
    ag = Agent(lay, pr)
    signal.signal(signal.SIGALRM, handler); signal.alarm(T)
    t0 = time.time()
    try:
        ag.learn_policy(T)
    except TimeoutException:
        print(f'seed {seed}: TIMEOUT -> run.py would score None'); return
    finally:
        signal.alarm(0)
    learn = time.time() - t0
    states = int(np.prod(ag.shape))
    v0 = ag.V[ag._encode(*env.get_state())]
    V_learned, pi_learned = ag.V, ag.policy
    delta = ag._sweep()                    # test-only: one more sweep tells us if V had converged
    converged = delta < 1e-6
    ag.V, ag.policy = V_learned, pi_learned  # evaluate exactly what learn_policy produced
    cap = 2 * env.N ** 2
    capped, full, steps = [], [], []
    for _ in range(episodes):
        env = TreasureHunt(lay, pr); s = env.get_state(); g = 1.0; rc = rf = 0.0; k = 0
        while not env.done and k < 20 * cap:
            s, r, d = env.step(ag.get_action(*s))
            if k < cap: rc += g * r
            rf += g * r; g *= env.df; k += 1
        capped.append(rc); full.append(rf); steps.append(k)
    full = np.array(full); se = full.std() / np.sqrt(episodes)
    ok = (not converged) or abs(full.mean() - v0) <= 3 * se + 0.02
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'seed {seed:3d}: N={N:2d} R1={s1:2d} R2={s2:2d} ps={ps:.2f} g={gamma:.2f} states={states/1e6:6.2f}M '
          f'learn={learn:5.1f}s/{T}s conv={"Y" if converged else "N"} V(start)={v0:7.3f} '
          f'MC={full.mean():7.3f}+-{1.96*se:.3f} capped={np.mean(capped):7.3f} steps={np.mean(steps):6.1f} '
          f'rss={rss:6.0f}MB {"OK" if ok else "MISMATCH"}')


if __name__ == '__main__':
    main()
