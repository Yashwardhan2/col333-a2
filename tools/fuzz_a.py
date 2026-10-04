import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brute_check_a import check
import tempfile
D = [(1, 0), (-1, 0), (0, -1), (0, 1)]
out = tempfile.mkdtemp(prefix='fuzz_a_')
os.makedirs(out, exist_ok=True)
rng = random.Random(1)
def blob(N, start, size, taken):
    reg = [start]; seen = {start}
    while len(reg) < size:
        i, j = rng.choice(reg); di, dj = rng.choice(D); n = (i + di, j + dj)
        if 0 <= n[0] < N and 0 <= n[1] < N and n not in seen and n not in taken: reg.append(n); seen.add(n)
    return reg
def gen(k):
    while True:
        N = rng.randint(4, 6)
        cells = [(i, j) for i in range(N) for j in range(N)]
        r1 = blob(N, rng.choice(cells), rng.randint(1, 4), set())
        moat = {(i + di, j + dj) for (i, j) in r1 for di, dj in D} | set(r1)
        free = [c for c in cells if c not in moat]
        if not free: continue
        r2 = blob(N, rng.choice(free), rng.randint(1, 3), moat)
        if any((i + di, j + dj) in set(r1) for (i, j) in r2 for di, dj in D): continue
        rest = [c for c in cells if c not in set(r1) | set(r2)]
        rng.shuffle(rest)
        if len(rest) < 5: continue
        g = {c: 'W' for c in cells}
        for c in r1: g[c] = '!'
        for c in r2: g[c] = '!'
        g[r1[0]] = '1'; g[r2[0]] = '2'
        g[rest[0]] = 'S'; g[rest[1]] = 'T'; g[rest[2]] = 'T'; g[rest[3]] = 'F'
        for c in rest[4:]:
            if rng.random() < 0.2: g[c] = 'L'
        if rng.random() < 0.3 and len(rest) > 6: g[rest[4]] = 'F'   # occasionally a second fort
        lay = f'{out}/{k}_layout.txt'; pr = f'{out}/{k}_prob.txt'
        open(lay, 'w').write('\n'.join(''.join(g[(i, j)] for j in range(N)) for i in range(N)) + '\n')
        ps = rng.choice([0.9, 0.6, 0.25, 0.2, 0.05])
        def pp():
            v = [rng.choice([0, 1, 2, 3]) for _ in range(4)]
            if sum(v) == 0: v[0] = 1
            return ' '.join(str(x / sum(v)) for x in v)
        rs = rng.choice(['-0.05 3 5 -1', '-0.1 1 2 -10', '-0.01 5 1 -3'])
        df = rng.choice([0.99, 0.9, 0.7])
        open(pr, 'w').write(f'{ps}\n{pp()}\n{pp()}\n{rs}\n{df}\n')
        return lay, pr, ps
worst_all = 0; bad_all = 0
for k in range(40):
    lay, pr, ps = gen(k)
    w, b, n = check(lay, pr)
    worst_all = max(worst_all, w); bad_all += b
    if w > 1e-6 or b: print('MISMATCH', k, ps, w, b)
print(f'40 random layouts: max|dV|={worst_all:.2e} suboptimal actions={bad_all}')
