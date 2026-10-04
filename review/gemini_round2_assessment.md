# Assessment of Gemini's round-2 review

**Status: no submission code has been changed.** Every claim was checked against the code (`part_a/agent.py`, `part_b/env.py`) or measured. Part B measurements use prototypes in `tools/experiments/part_b/`, which read simulator internals for measurement only; none of that goes into the agent. Section 4 lists the proposals for discussion.

---

## 1. Summary

| # | Gemini's claim | Verdict | Evidence |
|---|---|---|---|
| A1a | `get_action`'s fallback chain cannot crash (ship on land → KeyError → KeyError → `UP`) | **Correct** | Already tested in A6: invalid inputs return a legal action |
| A1b | "`eval_a.py` defines `class TO`", so a different exception name could slip through | **Out of date, but the concern is real** | `eval_a.py` was renamed to `TimeoutException` in A5; Gemini quoted the round-1 copy. A different class name *does* escape (tested), which is equivalent to having no safety net (not a regression). See P3. |
| A2 | Keep the 0.85·T guard, because an external `timeout T python run.py` could kill the process | **Right conclusion, wrong reason** | No external kill at T is possible, because T excludes the ~1000 evaluation episodes that run in the same process (TA clarification). The real reason: realistic grids converge long before T, even at T=1 (tested). |
| A3 | We are at "the floor"; the pre-solve takes "~5 ms"; buffer reuse would save only microseconds | **The conclusion holds; the specifics are inaccurate** | The pre-solve takes 2–35 ms, not 5. Fresh-memory cost per sweep is ≤ 0.1 ms (≤ 2% of a sweep), so buffer reuse isn't worth it. "Floor" is rhetoric; realistic grids already converge in < 1 s. |
| B4a | 30k steps/s gives 1.8M steps at T=60, "36 visits per pair, plenty of data" | **Overstated** | A lean Q-learning loop runs at 24k steps/s, about 1.2M steps under the 0.85·T guard. Visits are very skewed: only 179–527 of the 10,000 states get ≥ 180 visits in 300k steps, and the top 10% of states take 54–61% of visits. |
| B4b | ε = max(0.01, 1 − t/(0.8T)) "guarantees pure exploitation in the final moments" | **Fine as a starting schedule; the justification is irrelevant** | Evaluation always uses the greedy policy, so ε only shapes the training data. With this schedule: 3.5 at T=60, 4.7 at T=240 (TA baseline ~6). |
| B4c | Constant α=0.1 beats 1/(1+n)^0.6 | **Consistent with our data** (2 seeds) | At T=60: 3.74 (α=0.1, with the collision fix) vs 3.18 (1/(1+n)^0.6). Still to be tuned (0.05, 0.2, …). |
| B4d | Q₀=0 is "natively pessimistic" and stops the agent "constantly testing actions that lead to −5" | **The choice is fine; the reasoning is off** | Crashes during training cost nothing (only evaluation is scored), so avoiding exploration isn't a goal. Zero is simply the neutral init the TA rule allows. Note that an unvisited state's argmax is action 0 (speed up). |
| B5a | Delayed collision: the −5 comes one step late, Q(s_{t+1},·) "will become −5" and propagate back correctly | **Mechanism correct, conclusion false** | Verified in `env.step`: the collision check runs at the start of the next call, so the action there is irrelevant. But own-lane `min_dist`=1 observations are aliased: **72–80% of them are not collided** (a car 0.5–1.67 ahead, where swerving or braking matters). Q(s_{t+1},·) therefore never becomes −5, and the irrelevant −5s pollute exactly the states where evasive actions should be learned. A fix (crediting the −5 to the previous action) scores better at both T=60 and T=240 (section 3). |
| B5b | `min_dist`=1 aliases "cars safely behind you" with "cars dangerously close ahead" | **Wrong about "safely behind"** | `Lane.cars` is sorted and `remove_cars` drops cars more than 0.5 behind. So a "behind" reading means a car within 0.5 behind: a collision in our own lane, or a car alongside in another lane. Both cases are dangerous. The real aliasing in our own lane is "already collided" (20–28%) vs "car 0.5–1.67 ahead" (72–80%). |

---

## 2. Part A details

### 2.1 Different exception class (A1b)

Test: the 31.5M-state grid at T=5, with the alarm raising a dynamically named exception class.

| Class name | Result |
|---|---|
| `TimeoutException` (run.py's name) | `learn_policy` returns normally and evaluation proceeds |
| `AlarmTimeout` (any other name) | The exception escapes `learn_policy`, so `run.py` would print "exceeded" and score `None` |

This is exactly the pre-A5 behaviour, so nothing got worse. Piazza says evaluation uses `run.py` (with `visualize=False`), whose class is `TimeoutException`, so the risk is low. **P3** below would remove it entirely.

### 2.2 Alarm delay (correction to our own earlier number)

In the same test, the deferred alarm overran T by **2.3 s**, because one large `einsum` call was in progress. We previously reported "about 0.4 s"; the true bound is the length of the longest single numpy call, which is up to about 3 s on the 31.5M-state grid. On realistic grids these calls take milliseconds. **P4** below would shrink it on huge grids.

### 2.3 The time guard on a realistic worst case (A2)

Layout: 30×30 open water, two 6-cell pirate regions, targets in the far corners, γ=0.99, shape (4, 900, 6, 6).

- The pre-solve takes 22 ms, then 114 sweeps in 0.61 s reach `max|ΔV|` < 1e-9.
- At T=1, learning stops at 0.7 s with V(start) = 1.4044, the **same as at T=30**. Scores over 1000 runs: 1.38 ± 0.03 at T=1 and 1.41 ± 0.03 at T=30, equal within noise.
- T is an integer ≥ 1 (`signal.alarm`), so on realistic grids we converge at every possible budget, and loosening the guard can't help there.

### 2.4 Allocation cost (A3)

Measured with `ru_minflt` after a warm-up sweep:

| Layout | States | Sweep time | Page faults per sweep | Cost of that fresh memory |
|---|---|---|---|---|
| small | 0.08M | 3.2 ms | 408 | 0.1 ms (2%) |
| corridors | 2.9M | 159 ms | 298 | 0.2 ms (0%) |
| blocks | 31.6M | 5.0 s | 6,552 | 5.6 ms (0.1%) |

numpy and glibc reuse the freed blocks, so pre-allocating buffers would gain nothing measurable.

---

## 3. Part B details

### 3.1 Simulator facts verified in `part_b/env.py`

- **When the collision is detected.** `HighwayEnv.step` first checks for a collision at the *current* configuration (the one produced by the previous action). If there is one, it returns −5, `done=True` and the collided observation, without applying the new action.
- **Who gets the −5 by default.** Standard Q-learning therefore assigns the −5 to `(s_{t+1}, a_{t+1})`, an action that had no effect. The action that caused the crash, `a_t`, received its normal positive reward.
- **Which cars stay visible.** `Lane.cars` is kept sorted by position. `remove_cars` deletes cars with `pos + 0.5 < x`, so any car still behind us is within 0.5 behind.
- **What `min_dist`=1 covers.** It means `cars[0].pos < x`, or a car ahead at distance d < 1.67 (because `round(0.3·d + 1) = 1`).

### 3.2 Measurements (`tools/experiments/part_b/diag.py`)

- **Speed:** a lean ε-greedy Q-learning loop with a fresh `HighwayEnv()` per episode runs at **24k steps/s**. That is about 1.22M steps in 0.85·60 s and 4.9M in 0.85·240 s.
- **Coverage over 300k steps:**

  | Policy | Distinct observations seen | States with ≥ 180 visits | Share of visits in the top 10% of states |
  |---|---|---|---|
  | random | 6,678 | 179 | 54% |
  | ε=0.1 greedy | 5,504 | 527 | 61% |

- **Aliasing:**
  - All collisions are returned with own-lane `min_dist`=1.
  - Among non-terminal own-lane=1 observations, 20–28% are already-collided configurations (the next step is −5 whatever we do), and 72–80% have a car 0.5–1.67 ahead (the action still matters).
  - Among observations that are *sometimes* collided, 70–77% of visits are not collided.

### 3.3 Q-learning prototypes (`tools/experiments/part_b/proto_q.py`)

- Single core, γ=0.99, 300 greedy evaluation episodes in fresh environments, scored like `run.py`.
- **"gemini"** is Gemini's recipe: ε = max(0.01, 1 − t/(0.8T)), α = 0.1, Q₀ = 0, standard terminal update.
- **"+fix"** credits the −5 to the action that caused the crash. The update becomes Q(s_t,a_t) ← r_t + γ·r_{t+1}, and the collision step's own (s,a) is not updated. It uses only the observed reward and the `done` flag, with no reward constants.

| Variant | T | Seeds | Scores | Mean |
|---|---|---|---|---|
| gemini | 60 | 1, 2, 3 | 3.51, 3.35, 3.73 | 3.53 |
| gemini + fix | 60 | 1, 2, 3 | 3.77, 3.72, 3.72 | **3.74** |
| fix + α = 1/(1+n)^0.6 | 60 | 1, 2 | 3.13, 3.22 | 3.18 |
| gemini | 240 | 1, 2 | 4.80, 4.58 | 4.69 |
| gemini + fix | 240 | 1, 2 | 4.88, 4.89 | **4.88** |

**Takeaways:**
- The fix gives about +0.2 at both budgets and is much more consistent across seeds.
- Constant α beats the visit-count α.
- **Neither variant reaches the TA baseline of ~6 at T=240.** Gemini's recipe is a starting point, not a solution; Part B needs real tuning.
- With the fix, the crash rate is higher (22–25% vs 12%) but so is the score. It drives faster, so the speed/safety trade-off is worth studying.

---

## 4. Proposals (for discussion, nothing implemented)

**Part A (both optional, low priority; they only matter on unrealistic huge grids or a renamed exception)**

- **P3: broaden the safety net.** Also treat these as "timeout, keep the last complete policy":
  - any exception raised once the elapsed time is at least about T (whatever its class name);
  - a `MemoryError` at any time.

  *Risk:* it hides a genuine bug that fires after T. Our validation tools compare V against the brute-force reference, so a bug would still be caught there.
- **P4: per-mask sweep.** Process the four treasure masks one at a time inside each sweep (same Jacobi maths, so identical results), and check the deadline between masks.
  - It cuts the longest single numpy call, and with it the alarm overrun, by about 4×.
  - It lowers peak memory.
  - It lets a sweep that is out of time stop early while keeping the previous policy.

  *Cost:* a little Python overhead per sweep, and slightly more code.

**Part B (design inputs for B3/B4, to adopt when we write `part_b/agent.py`)**

- **B-i:** use the collision-credit fix. Treat every `done` as "the last reward belongs to the previous action". That even avoids relying on the 1000-step cap: at a truncation the mis-credit happens once per 1000 steps, which is harmless.
- **B-ii:** start from constant α = 0.1, ε linear from 1 to 0.01 over 0.8·T, Q₀ = 0 and `discount_factor` taken from the constructor. Then tune α, the ε floor and length, and the speed/safety trade-off against the ~6 target at T=240, also checking T=60 and other γ.
- **B-iii:** pick a deliberate default action for states never visited (currently argmax of zeros, i.e. speed up), for example the most common greedy action among visited states with the same speed and lane.
