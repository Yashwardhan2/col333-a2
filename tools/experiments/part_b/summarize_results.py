"""Summarise eval_b.py / episode_stats.py result lines grouped by label prefix.

usage: python summarize_results.py <results.txt>
Lines from eval_b.py look like:
  '<label> T=60 df=0.99: learn 51.0s, 1.20M steps (23.5k/s) | score 4.05 +- 0.11 (300 eps) | crash rate 0.15 | mean length 900 | mean speed idx 0.34 | ...'
"""
import re, sys
from collections import defaultdict
import numpy as np

EVAL = re.compile(r'^(?P<label>\S+) T=(?P<T>\d+) df=(?P<df>[\d.]+): learn [\d.]+s, (?P<steps>[\d.]+)M steps .*?\| score (?P<score>-?[\d.]+) \+- [\d.]+ .*?\| crash rate (?P<crash>[\d.]+) .*?mean speed idx (?P<speed>[\d.]+)')
STATS = re.compile(r'^(?P<label>\S+) T=(?P<T>\d+) df=(?P<df>[\d.]+): discounted (?P<disc>-?[\d.]+) \+- [\d.]+ = driving (?P<drive>-?[\d.]+) \+ crash penalty (?P<pen>-?[\d.]+) \| undiscounted (?P<und>-?[\d.]+) .*?crash rate (?P<crash>[\d.]+), crash step p10/25/50/75/90 = (?P<q>\[[^\]]*\]) \| mean speed idx (?P<speed>[\d.]+)')

groups = defaultdict(list)
stats = defaultdict(list)
for line in open(sys.argv[1]):
    line = line.strip()
    m = EVAL.match(line)
    if m:
        d = m.groupdict()
        groups[(d['label'], int(d['T']), float(d['df']))].append((float(d['score']), float(d['crash']), float(d['speed']), float(d['steps'])))
        continue
    m = STATS.match(line)
    if m:
        d = m.groupdict()
        stats[(d['label'], int(d['T']), float(d['df']))].append(d)


def fmt(rows):
    s = np.array([r[0] for r in rows])
    return (f'{s.mean():7.3f} (sd {s.std(ddof=1) if len(s) > 1 else 0:.3f}, n={len(s)}) '
            f'crash {np.mean([r[1] for r in rows]):.2f} speed {np.mean([r[2] for r in rows]):.2f} steps {np.mean([r[3] for r in rows]):.2f}M')


def section(prefix, title):
    keys = sorted((k for k in groups if k[0].startswith(prefix)), key=lambda k: (k[1], k[2], k[0]))
    if not keys:
        return
    print(f'\n== {title}')
    for k in keys:
        print(f'  {k[0]:28s} T={k[1]:3d} df={k[2]:<6}: {fmt(groups[k])}')


section('G:', 'gamma matrix (claim 3): tuned vs constant alpha')
section('M:', 'mechanism (claim 2), T=60 gamma=0.99')
section('B5:', 'B5 stop fraction (claim 4): 0.85T vs 0.95T with safety net')
section('B6:', 'B6 small-budget / gamma sweep (C current, E step-eps, A alpha 500k)')
if stats:
    print('\n== per-episode statistics (claim 1)')
    for k, rows in sorted(stats.items()):
        for d in rows:
            print(f"  {k[0]:10s} T={k[1]} df={k[2]}: discounted {d['disc']} = driving {d['drive']} + crash {d['pen']} | "
                  f"undiscounted {d['und']} | crash rate {d['crash']} | crash-step quantiles {d['q']} | speed {d['speed']}")
