"""Explain COL333-A2-CHECKER verdicts that are not 'matched' (Part A).

usage: python tools/analyze_checker_a.py <checker dir> <log> [<log> ...]
  <log> is the saved output of `benchmark.py evaluate --part a ...` (one or more suites).
For every 'suboptimal' / 'more optimal' / 'error' line it re-solves the case and reports
whether our VI converged, ties at the start state, and what the 2N^2 cap costs.
"""
import os, sys, re, json, tempfile, time
from pathlib import Path
import numpy as np
CHECKER = Path(sys.argv[1]).resolve()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments'))
sys.path.insert(0, str(CHECKER))
from proto_variants import Proto
import benchmark as bm

line_re = re.compile(r'^(suite_\d+-\d+) (matched|more optimal|suboptimal|error) score=(\S+) reference=(\S+)')
cases = {}
for s in ('suite_001', 'suite_002', 'suite_003'):
    for c in json.load(open(CHECKER / 'model_scores' / 'part_a' / f'{s}.json'))['cases']:
        cases[c['id']] = c
for log in sys.argv[2:]:
    for line in open(log):
        m = line_re.match(line.strip())
        if not m or m.group(2) == 'matched':
            continue
        cid, status, score, ref = m.group(1), m.group(2), m.group(3), m.group(4)
        case = cases[cid]
        with tempfile.TemporaryDirectory() as d:
            lay, pr = bm._write_part_a_files(case, Path(d))
            ag = Proto(str(lay), str(pr))
            t0 = time.time(); ag.learn_policy(10.0); tl = time.time() - t0
            s0 = ag._encode(*bm._TreasureHunt(str(lay), str(pr)).get_state())
            V_learned, pi = ag.V.copy(), ag.policy.copy()
            delta = ag._sweep()
            scale = max(1.0, float(np.max(np.abs(V_learned))))
            converged = delta < 1e-6 * scale
            # ties at the start state: Q of every action
            W = ag._pirate_expectation(ag._landing_values(V_learned))
            q = (1 - ag.ps) / 3; c = ag.ps - q
            G = np.array([W[(s0[0], ag.nxt[s0[1], d], s0[2], s0[3])] for d in range(4)])
            Q = q * G.sum() + c * G
            ties = int(np.sum(Q >= Q.max() - 1e-9 * scale))
            # value of our stationary policy under the 2N^2 cap vs the best time-aware policy
            H = 2 * ag.N ** 2; allm = list(range(ag.n_masks))
            if H * np.prod(ag.shape) > 3e8:
                print(f'{cid} {status:12s} score={float(score):10.4f} ref={float(ref):10.4f} | N={ag.N} gamma={case["prob"]["discount"]} '
                      f'ps={case["prob"]["ship_success"]} learn={tl:4.1f}s converged={converged} start-state ties={ties} | '
                      f'our V(start)={V_learned[s0]:.4f} (cap analysis skipped: too large)', flush=True)
                continue
            Vp = np.zeros(ag.shape)
            for _ in range(H):
                Vp = ag.ship_fixed(ag._pirate_expectation(ag.landing(Vp, allm)), ag.nxt, pi)
            ag.V = np.zeros(ag.shape)
            for _ in range(H):
                ag._sweep()
            g, ps = case['prob']['discount'], case['prob']['ship_success']
            print(f'{cid} {status:12s} score={float(score):10.4f} ref={float(ref):10.4f} | N={ag.N} gamma={g} ps={ps} '
                  f'learn={tl:4.1f}s converged={converged} start-state ties={ties} | '
                  f'our V(start) inf-horizon={V_learned[s0]:.4f}  capped V^pi_H={Vp[s0]:.4f}  best time-aware V*_H={ag.V[s0]:.4f} '
                  f'(cap costs us {ag.V[s0]-Vp[s0]:.4f})', flush=True)
