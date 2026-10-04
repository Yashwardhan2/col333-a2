"""Why does the agent crash? Train the current part_b agent for T seconds, then run greedy
episodes while recording simulator internals (MEASUREMENT ONLY, never used by the agent).

usage: python crash_causes.py <T> <episodes>
For each collision it classifies the configuration that collided (the one produced by the
previous step, since env.step detects a collision before acting):
  we changed lane     our lane changed during the previous step
  it changed lane     the other car changed lane during the previous step
  we caught up        same lane, other car ahead, we were faster
  rear-ended          same lane, other car behind and faster (the mechanism claimed in review)
It also checks the claim that faster cars can approach from behind: it counts, at every
step, cars that are more than 0.5 behind the control car (remove_cars should delete them).
"""
import os, sys, time
from collections import Counter
import numpy as np
PB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'A2-starter-code', 'A2-starter-code', 'part_b')
sys.path.insert(0, PB)
from env import HighwayEnv
from agent import Agent

T, episodes = float(sys.argv[1]), int(sys.argv[2])
agent = Agent(HighwayEnv(), discount_factor=0.99)
agent.learn_policy(T)


def snapshot(env):
    cars = {}
    for L in env.lanes:
        for c in L.cars:
            cars[id(c)] = (L.lane_id, c.pos, c.speed)
    cc = env.control_car
    return cc.pos, cc.lane_id, cc.speed, cars


causes, crash_speed, far_behind, unvisited_before_crash = Counter(), Counter(), 0, 0
steps_total, crashes, speed_hist = 0, 0, Counter()
for _ in range(episodes):
    env = HighwayEnv(); s = env.get_state(); hist = []; recent_unvisited = []
    while not env.done:
        snap = snapshot(env)
        x = snap[0]
        far_behind += sum(1 for (_, p, _) in snap[3].values() if p < x - 0.5)
        recent_unvisited = (recent_unvisited + [sum(agent.visits[agent._index(*s)]) == 0])[-5:]
        a = agent.get_action(*s)
        hist.append((snap, a))
        speed_hist[snap[2]] += 1
        s, r, d = env.step(a); steps_total += 1
        if d and r < 0:
            crashes += 1
            unvisited_before_crash += any(recent_unvisited)
            (x, lane, spd, cars) = snap                 # the collided configuration
            prev = hist[-2][0] if len(hist) >= 2 else None
            crash_speed[spd] += 1
            close = [(abs(p - x), cid, p, sp) for cid, (l, p, sp) in cars.items() if l == lane and abs(p - x) < 0.5]
            if not close:
                causes['no car within 0.5 in our lane?'] += 1; continue
            _, cid, p, sp = min(close)
            if prev is not None and prev[1] != lane:
                causes['we changed lane into it'] += 1
            elif prev is not None and cid in prev[3] and prev[3][cid][0] != lane:
                causes['it changed lane into us'] += 1
            elif prev is not None and cid not in prev[3]:
                causes['car appeared (spawned)'] += 1
            elif p >= x:
                causes['we caught up (car ahead, same lane)'] += 1
            elif prev is not None and prev[3][cid][1] >= prev[0]:
                # it was ahead of us one step earlier and is now just behind: we drove
                # through it in one step (we move speed*0.3 per step)
                causes['we caught up and passed through it (car was ahead one step earlier)'] += 1
            else:
                causes['rear-ended by a car that was behind us'] += 1

print(f'trained {T:.0f}s ({agent.stats["steps"]/1e6:.2f}M steps); {episodes} greedy episodes, {steps_total} steps, {crashes} crashes')
print('crash causes:', dict(causes.most_common()))
print('our actual speed at the crash:', dict(sorted(crash_speed.items())))
print('time spent at each actual speed:', {k: f'{100*v/steps_total:.1f}%' for k, v in sorted(speed_hist.items())})
print(f'cars more than 0.5 behind us observed at any step: {far_behind} (remove_cars deletes them)')
print(f'crashes with an unvisited state among the last 5 steps: {unvisited_before_crash}')
