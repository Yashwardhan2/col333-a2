"""B6: small-budget / gamma sweep of the submitted agent plus two test-only variants.

usage: python tools/experiments/part_b/b6/run_b6.py <out.txt> [workers=3]
Arms (labels B6:<arm>):
  C  submitted agent.py, unchanged
  E  b6/agent.py with step-based epsilon: 1 -> 0.15 over 150k env steps (Gemini round-3 idea, with a floor)
  A  submitted agent.py with alpha_decay_steps=500000 (Gemini round-3 idea)
Jobs are interleaved (arm innermost) so machine-load drift hits all arms alike.
"""
import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
EVAL = os.path.join(ROOT, 'tools', 'eval_b.py')
B6 = os.path.dirname(os.path.abspath(__file__))
ARMS = {
    'C': {},
    'E': {'AGENT_DIR': B6, 'AGENT_SET': 'eps_decay_steps=150000,eps_floor=0.15'},
    'A': {'AGENT_SET': 'alpha_decay_steps=500000'},
}
TS, GAMMAS, RUNS, EPISODES = [5, 10, 20, 30, 60], [0.9, 0.97, 0.99, 0.999], 3, 300

out = sys.argv[1]
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
jobs = [(T, g, arm) for _ in range(RUNS) for T in TS for g in GAMMAS for arm in ARMS]


def run(job):
    T, g, arm = job
    env = dict(os.environ, **ARMS[arm])
    r = subprocess.run([sys.executable, EVAL, str(T), str(EPISODES), str(g), f'B6:{arm}'],
                       env=env, capture_output=True, text=True)
    line = (r.stdout.strip() or f'B6:{arm} T={T} df={g}: ERROR {r.stderr.strip()[-300:]}')
    with open(out, 'a') as f:
        f.write(line + '\n')
    return line


with ThreadPoolExecutor(workers) as ex:
    for i, line in enumerate(ex.map(run, jobs), 1):
        print(f'[{i}/{len(jobs)}] {line}', flush=True)
