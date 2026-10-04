"""Convergence + anytime comparison of the suggested variants (single core)."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto_variants import Proto, run_variant

layout, prob = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
variants = sys.argv[3].split(',')
fracs = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

run_variant(layout, prob, 'zeros')   # warm-up run (page faults, caches), discarded
ref_ag, ref_snaps, ref_n, ref_t = run_variant(layout, prob, 'zeros')
Vstar = ref_ag.V
s0 = ref_ag.start_index(layout)
print(f'{os.path.basename(layout)} shape={ref_ag.shape}  V*(start)={Vstar[s0]:.4f}  '
      f'baseline: {ref_n} sweeps, {ref_t:.3f}s to max|dV|<1e-9')
print('anytime: V^pi(start) of the greedy policy at fractions of the BASELINE convergence time')
print(f'{"variant":16s} {"sweeps":>6s} {"t_conv":>7s} {"t(|V-V*|<1e-4)":>15s} {"final err":>9s} | ' +
      ' '.join(f'{f:>6.2f}' for f in fracs))
cache = {}
for v in variants:
    ag, snaps, n, t = (ref_ag, ref_snaps, ref_n, ref_t) if v == 'zeros' else run_variant(layout, prob, v)
    err = float(np.max(np.abs(ag.V - Vstar)))
    t_close = next((ts for ts, V in snaps if np.max(np.abs(V - Vstar)) < 1e-4), float('nan'))
    row = []
    for f in fracs:
        tt = f * ref_t
        cand = [V for ts, V in snaps if ts <= tt]
        if not cand: row.append('   n/a'); continue
        V = cand[-1]
        key = (v, len(cand))
        if key not in cache:
            _, pi = ag.greedy(V)
            cache[key] = ag.evaluate(pi)[s0]
        row.append(f'{cache[key]:6.3f}')
    print(f'{v:16s} {n:6d} {t:7.3f} {t_close:15.3f} {err:9.1e} | ' + ' '.join(row))
