# Assessment of Gemini's Part B review (round 1)

**Status: `part_b/agent.py` has not been changed.** Every claim was checked against `part_b/env.py` and/or measured.
- The measurement tools are `tools/eval_b.py` (run.py-equivalent; `AGENT_SET` overrides settings for experiments) and `tools/experiments/part_b/crash_causes.py`. The latter reads simulator internals for measurement only.
- Single core, γ=0.99, 300 greedy evaluation episodes per run unless stated otherwise.
- Section 4 lists proposals for discussion.

---

## 1. Summary

| # | Gemini's claim | Verdict | Evidence |
|---|---|---|---|
| 1 | "Crashing *because* it drives too slowly: faster cars from behind plow into it; speed ≥ 2 is a survival necessity" | **False, and the opposite is true** | No car is ever more than 0.5 behind us (0 in 345k steps), because `remove_cars` deletes them and new cars spawn only ahead. **Zero crashes at speed 1**; most happen at speeds 3–4, where the agent spends only 1–4% of its time. |
| 2 | The collision credit already isolates the safe 72–80% of `min_dist`=1 observations; no representation change needed | **Correct** (this restates our B3 design) | The collided observation is neither updated nor bootstrapped from. The coarse distance bins and unobserved car speeds still leave aliasing, which is part of why high speed is hard (section 3). |
| 3 | Decay ε to the floor much earlier (by 0.2–0.3·T), because "random exploratory crashes depress Q-values of safe states" | **False. Measured: clearly worse** | 0.8·T: **3.87 / 5.21** (T=60 / T=240). 0.3·T: 3.31 / 4.46. 0.2·T: 2.70 / 4.01. Longer exploration is better at both budgets. The mechanism is also wrong for Q-learning (section 2.3). |
| 4a | `Q(s_t,a_t) ← r_t + γ·r_{t+1}` is exact for any γ | **Correct** | After a collision the episode ends, so the sampled return is exactly r_t + γ·r_{t+1}. With γ=0 it reduces to r_t. |
| 4b | One fast ε schedule will suffice for every γ | **Untested** | To be checked in B6 across γ ∈ {0.5 … 0.999}. Given claim 3's result, a fast schedule is the wrong starting point. |
| 5a | Default to no-op when a state's Q-values are all zero; argmax→0 (speed up) is "dangerous" | **Plausible but small; test it, don't assume it** | Never-updated states are 0.01–0.05% of evaluation steps. 2/44 and 7/82 crashes had one among the last 5 steps, but that is association, not causation (states next to collisions are rare by nature). |
| 5b | Dropping one bootstrap at the 1000-step limit is negligible | **Agree** | One update per surviving episode. |
| 5c | ε recomputed every 64 steps is safe | **Agree** | |

---

## 2. Details

### 2.1 Claim 1: "slow driving gets you rear-ended" (false)

**From the code (`env.py`):**
- `HighwayEnv.step` → `remove_cars(x)` → `Lane.remove_cars` deletes every car with `pos + 0.5 < x`, i.e. more than 0.5 behind the control car, after every step.
- `generate_cars` only creates cars ahead (at x+4 or x+10, or 10 behind the last car).
- `change_lanes` only moves cars with `pos ≥ x + 2`.
- Other cars' speeds are in [1, 2], and `modulate_speeds` only ever lowers a car to the speed of the car ahead of it (≥ 1).

So nothing can approach from behind, and at speed 1 we can't catch up with anything either.

**Measured** (`crash_causes.py`: two independently trained agents at T=60, 400 greedy episodes each):

| | Agent 1 | Agent 2 |
|---|---|---|
| Crashes | 44 | 82 |
| We caught up with a car ahead in our lane | 37 | 63 |
| We drove *through* a slower car in one step (it was ahead one step earlier; at speed 4 we move 1.2 per step) | 7* | 8 |
| We changed lane into a car | 0 | 11 |
| Genuinely rear-ended by a car behind | 0 | **0** |
| Our speed at the crash (1/2/3/4) | 0 / 8 / 15 / 21 | **0** / 15 / 26 / 41 |
| Share of time at speed 1/2/3/4 | 82.6 / 16.1 / 0.7 / 0.5 % | 72.6 / 23.4 / 3.0 / 1.1 % |
| Cars more than 0.5 behind us, over all steps | 0 | 0 |

\*Agent 1 ran before the "passed through" category existed, so its 7 were labelled "rear-ended". Agent 2's classifier separates them, and it found no genuine rear-ends.

**Conclusion:** crashes happen at speeds 2–4, overwhelmingly at 3–4, which the agent visits only 1–4% of the time. Speed 1 is perfectly safe but earns only 0.03 per step. The agent's problem is an **under-trained policy at high speed**, not slowness.

(My own earlier guess, that most crashes are lane changes, was also wrong: lane changes are 0–13% of crashes.)

### 2.2 Claim 2: aliasing (correct)

The collision credit already handles the collided 20–28% of own-lane `min_dist`=1 observations. What remains:
- The non-collided bins are coarse: bin 1 is d < 1.67, bin 2 is 1.67–5, bin 3 is 5–8.3, bin 4 is ≥ 8.3.
- Other cars' speeds aren't observed.
- At speed 4 we close on a speed-1 car by 0.9 per step. Bin 2 can therefore be crossed in about 2–4 steps, while braking from speed 4 to 1 needs 3 successful actions, each succeeding with probability 0.8.

So high speed is inherently risky under this observation. Learning *when* it is safe needs a lot of data in exactly the states the agent rarely visits.

### 2.3 Claim 3: decay ε earlier (false)

**Measured** (ε decays linearly from 1 to 0.01, reaching it at the given fraction of T; otherwise identical):

| ε reaches 0.01 at | T=60, 3 runs | Mean | T=240, 2 runs | Mean |
|---|---|---|---|---|
| **0.8·T (current)** | 3.81, 3.90, 3.91 | **3.87** | 5.27, 5.16 | **5.21** |
| 0.3·T | 3.18, 3.37, 3.38 | 3.31 | 4.72, 4.19 | 4.46 |
| 0.2·T (Gemini's suggestion) | 2.32, 2.65, 3.13 | 2.70 | 4.09, 3.93 | 4.01 |

Faster decay loses 0.6–1.2 points at both budgets, and the trend is monotone in favour of *more* exploration.

**Why the mechanism is wrong:** Q-learning is off-policy. Its target `r + γ·max_a' Q(s',a')` doesn't depend on which (possibly random) action the behaviour policy takes next. So random crashing actions lower only the Q-value of the random action itself, not the values of the safe greedy actions. What exploration *does* buy is coverage of rarely visited states, such as high speed, which section 2.1 shows is exactly where the agent is weak.

### 2.4 Claim 5a: default action for never-updated states (small, test it)

- Never-updated states make up 0.01–0.05% of evaluation steps.
- 2 of 44 and 7 of 82 crashes had one among their last 5 steps. That doesn't show the default action caused the crash, because configurations next to a collision are rare in themselves.
- A neutral default ("no-op", i.e. keep the current speed and lane), chosen by `visits == 0` rather than by Q being exactly zero, is cheap. Its expected effect is at most about 0.1 points.
- It should be decided by an A/B test in B4, not adopted on argument.

---

## 3. What this means for B4 (new evidence, not from the review)

- **Learn high-speed states better.** The score is capped by low speed, and the crashes are concentrated at speeds 3–4. More exploration helps monotonically, so the next things to test go in the *opposite* direction from Gemini's suggestion:
  - ε reaching its floor later (1.0·T);
  - a higher ε floor (0.05 or 0.1);
  - possibly a slower decay at small T.
- **Collect more data per second.** Throughput is about 24–26k steps/s. Cheaper per-step code (flat indices, pre-drawn random numbers) gives more updates within the same T. This also helps the 20% graded at small budgets.
- **α:** constant 0.1 is the current setting. Try 0.05–0.2, since more data per state may favour a smaller final α.
- **Noise discipline:** single runs vary by about ±0.15, and replicates at T=240 differed by up to 0.3 (for example, 0.8·T scored 4.95/4.67 earlier and 5.27/5.16 here). Use at least 3 replicates per setting at T=240.

---

## 4. Proposals (for discussion, nothing implemented)

1. **Keep ε reaching its floor at 0.8·T** (reject Gemini's 0.2–0.3·T). In B4, test 1.0·T and floors of 0.05 and 0.1.
2. **No-op default for never-updated states:** A/B test it in B4 (low expected impact). Adopt it only if it doesn't hurt.
3. **B4 priority order:**
   - (a) ε schedule and floor (longer exploration);
   - (b) throughput;
   - (c) α;
   - (d) the no-op default;

   each with at least 3 replicates at T=60 and T=240.
