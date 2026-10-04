# Assessment of Gemini's review (Part A, steps A1–A4, plus the Part B plan)

**Status: no submission code has been changed.** `part_a/agent.py` is exactly as reviewed. Every claim below was checked against `env.py`, against exact computations, or against measurements made with prototype code in `tools/experiments/`. Section 4 lists the changes I propose, for us to discuss before anything is implemented.

---

## 1. Summary

| # | Gemini's claim or suggestion | Verdict | Proposed action |
|---|---|---|---|
| 1 | Model and order of operations match `env.py` | **Correct.** We had already verified this independently (brute-force, fuzzing, Monte-Carlo). | None |
| 1b | Pirate start cells (`1`, `2`) must be legal sea cells | **Correct**, and already handled. The brute-force check uses `env.ship_loc_validity` itself. | None |
| 2a | Layered solving by treasure mask is "critical" | **Wrong as stated.** It needs 1.9–2.7× more sweeps, is slower in 3 of 4 layouts, and its anytime quality is the **worst** of all variants. | Do not adopt. Remove it from the A5 plan. |
| 2b | BFS-distance initialization of the m=0 layer | **Right idea, weak formula.** No measurable gain on 3 of 4 layouts. | Replace with a relaxed pirate-free pre-solve (**P1**) |
| 2c | Gauss–Seidel in-place updates | **Fewer sweeps but slower in wall-clock time** with numpy (1.2–6× slower). A per-cell Python loop would be slower still. | Do not adopt |
| 3 | Sparse 5-point stencil for the pirate expectation | **Correct maths, 2–3× slower** than our einsum, and it needs 5× more memory. | Do not adopt |
| 4 | Ignore the 2N² cap and stay infinite-horizon | **Right conclusion, wrong reasoning.** The exact gain from a time-aware policy is ≤ 1e-9 on all public tests. | None |
| 5a | "Q-table (4,4,5,5,5,5,5) has 100,000 states" | **Wrong.** It is 10,000 states × 5 actions = 50,000 entries. | None (our plan was already right) |
| 5b | Optimistic Q init of 5.0 | **Conflicts with a TA rule** (no reward knowledge, even at initialization). It is also not optimistic at γ=0.99 and far too optimistic at γ=0.9. | Use only optimism derived from data, and test it |
| 5c | Decay ε by elapsed fraction of T | **Correct.** Already in our plan. | None |
| 5d | Fresh `HighwayEnv()` per episode | **Correct.** Already in our plan. Costs 0.25 ms per episode. | None |

---

## 2. Part A: details and evidence

**How things were measured**
- Single core, `OPENBLAS_NUM_THREADS=1`, numpy 2.2.6.
- Each comparison ran on its own core of a shared 4-core container, after a discarded warm-up run.
- Wall-clock times are noisy: the same work can differ by up to about 1.5× between runs. Sweep counts are exact.
- **Anytime quality** means the exact value V^π(start) of the greedy policy at a given moment, computed by exact policy evaluation (not sampling). It is measured at a fraction of the *current* agent's convergence time.
- Every variant converges to the same V* (final error ≤ 1e-8), so all differences are in speed only.

**Layouts**
- Test 1 and test 2.
- **small:** 30×30 with 15% land and two 5-cell pirate regions.
- **maze:** a 30×30 serpentine maze with a path of about 450 steps and two 4-cell pirate regions.
- **blocks:** 30×30 with two 100-cell pirate regions (31.5M states).

All are in `tools/experiments/layouts/`.

### 2.1 Q1: model consistency, edge cases (agree)

All of Gemini's points here are correct. Our own checks are stronger than its reading of the code:

- **Brute-force cross-check:** an unfactored MDP built from `env.py`'s own functions matches our values to ≤ 4e-9, with 0 suboptimal actions, on all 3 tests.
- **Fuzzing:** 40 random layouts, including ps < 0.25, two forts, zero-probability pirate moves and γ ∈ {0.7, 0.9, 0.99}, match too.
- **Monte-Carlo:** returns sampled from the real `env.step` match V(start) within the confidence interval.

One small correction: Gemini says we relied on `(1-ps)/3` "rather than checking env.py". We did check `env.py`. It uses `random.random() > ps` and then picks uniformly from the other 3 actions with `np.random.choice`. Its pirate-start-cell point is also already covered: those cells are non-land, and the brute-force check enumerates ship states with `env.ship_loc_validity`.

### 2.2 Q2a: layered solving by treasure mask (disagree)

Gemini's argument is that solving m=0 first "cuts the active state space by 75% during the hardest phase". The measurements contradict it:

| Layout | Sweeps (current → layered) | Time to converge (current → layered) | V^π(start) at 50% of current time (current / layered) |
|---|---|---|---|
| test 1 | 86 → 161 | 0.051 → 0.035 s | 5.57 / −2.89 |
| test 2 | 193 → 457 | 0.239 → 0.389 s | 1.75 / **−5.00** |
| small 30×30 | 138 → 280 | 0.487 → 0.748 s | 1.16 / **−4.86** |
| maze 30×30 | 268 → 727 | 0.397 → 0.810 s | −3.53 / **−5.00** |

Why it loses:
1. Each layer pays its own convergence tail, the slow γ-contraction to the tolerance, so the total number of sweeps grows by 1.9–2.7×.
2. The **start state is in the last layer** (both treasures present). Until that layer is solved, the policy at the start state knows nothing, and it is stuck at −5 on 3 of 4 layouts. This is the worst possible behaviour for the 20% graded on small time budgets.

Layered solving combined with P1 is also no better than P1 alone (see the tables in 2.4).

### 2.3 Q2b: BFS-distance initialization (agree with the idea, not the formula)

Gemini's formula is `V_init = R_step·dist + R_fort` for the m=0 layer only, with other layers left at 0. It has three problems:
- It ignores discounting.
- It ignores wind.
- It gives no information about treasures, while the start state is in the m=3 layer.

**Result:** sweep counts 86→79, 193→188, 138→131 and 268→268. Anytime quality is essentially unchanged from the current agent (tables in 2.4).

A much better version of the same idea is **P1** (section 4): solve the small pirate-free MDP over (mask, cell) exactly, with wind, discounting, treasures and fort included, and use that as the initial V. It costs only milliseconds.

### 2.4 Q2c: Gauss–Seidel (disagree for a numpy implementation)

I tested the strongest practical version: cells are grouped into bands by BFS distance from the fort and treasures, the bands are processed outward, and each band is vectorized. A literal per-cell Python loop, as Gemini suggests, has strictly more overhead: 400–900 Python iterations per sweep instead of a few dozen to a few hundred bands.

Full anytime tables follow. Each cell is V^π(start) at the given fraction of the *current* agent's convergence time. "n/a" means the P1 pre-solve (2–35 ms) had not finished yet at that point.

**Test 1** (V* = 6.068; current agent: 86 sweeps, 0.051 s)

| Variant | Sweeps | Time to converge | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|---|
| current (V=0) | 86 | 0.051 s | −4.99 | −4.93 | 5.27 | 5.57 |
| layered | 161 | 0.035 s | −4.32 | −4.32 | −4.31 | −2.89 |
| Gemini BFS init | 79 | 0.032 s | −4.97 | −2.10 | 5.57 | 6.07 |
| **P1 relaxed init** | 76 | 0.038 s | n/a | **6.07** | **6.07** | **6.07** |
| layered + P1 | 140 | 0.036 s | n/a | 6.01 | 6.01 | 6.06 |
| band GS | 50 | 0.098 s | −1.68 | 2.97 | 3.20 | 5.31 |
| band GS + P1 | 46 | 0.085 s | n/a | 6.00 | 6.07 | 6.07 |

**Test 2** (V* = 1.747; current agent: 193 sweeps, 0.239 s)

| Variant | Sweeps | Time to converge | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|---|
| current (V=0) | 193 | 0.239 s | −5.00 | −3.47 | 1.43 | 1.75 |
| layered | 457 | 0.389 s | −5.00 | −5.00 | −5.00 | −5.00 |
| Gemini BFS init | 188 | 0.402 s | −5.00 | −5.00 | −2.91 | 1.72 |
| **P1 relaxed init** | 135 | **0.138 s** | **1.74** | **1.75** | **1.75** | **1.75** |
| layered + P1 | 386 | 0.399 s | 1.45 | 1.45 | 1.45 | 0.13 |
| band GS | 94 | 0.376 s | −2.75 | −2.91 | −2.90 | 1.75 |
| band GS + P1 | 74 | 0.314 s | 1.45 | 1.36 | 1.75 | 1.75 |

**small 30×30** (V* = 1.571; current agent: 138 sweeps, 0.487 s)

| Variant | Sweeps | Time to converge | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|---|
| current (V=0) | 138 | 0.487 s | −5.00 | −5.00 | −4.37 | 1.16 |
| layered | 280 | 0.748 s | −4.86 | −4.86 | −4.86 | −4.86 |
| Gemini BFS init | 131 | 0.623 s | −4.97 | −5.00 | −5.00 | 0.15 |
| **P1 relaxed init** | 85 | **0.263 s** | **1.57** | **1.57** | **1.57** | **1.57** |
| layered + P1 | 173 | 0.246 s | 1.57 | 1.57 | 1.57 | 1.57 |
| band GS | 67 | 0.563 s | −5.00 | −2.95 | −2.95 | 1.57 |
| band GS + P1 | 45 | 0.470 s | 1.57 | 1.57 | 1.57 | 1.57 |

**maze 30×30** (V* = −3.530; current agent: 268 sweeps, 0.397 s)

| Variant | Sweeps | Time to converge | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|---|
| current (V=0) | 268 | 0.397 s | −5.00 | −5.00 | −5.00 | −3.53 |
| layered | 727 | 0.810 s | −5.00 | −5.00 | −5.00 | −5.00 |
| Gemini BFS init | 268 | 0.397 s | −5.00 | −5.00 | −5.00 | −3.53 |
| **P1 relaxed init** | 252 | 0.353 s | n/a | **−3.53** | **−3.53** | **−3.53** |
| layered + P1 | 665 | 0.539 s | n/a | −3.53 | −3.53 | −3.53 |
| band GS | 224 | 2.494 s | −5.00 | −5.00 | −5.00 | −5.00 |
| band GS + P1 | 128 | 1.445 s | n/a | −3.53 | −3.53 | −3.53 |

**Conclusions**
- Band Gauss–Seidel cuts sweeps by 1.2–2.1×. But each banded sweep costs 2–8× more in numpy, because of Python per-band overhead and the recomputation of the landing and pirate stages on overlapping cells. So it is never faster in wall-clock time.
- Band GS combined with P1 is always slower than P1 alone.
- **P1 is the clear winner on every layout.** It reaches the optimal policy at 5–10% of the current agent's convergence time and converges 1.1–1.9× faster.

**Worst case, blocks layout (31.5M states, about 7.8 s per sweep), T = 60 s, 200 evaluation runs:**

| Agent | Full sweeps done | Score |
|---|---|---|
| current | 7 | −5.000 ± 0.000 |
| with P1 | 6 | **+1.900 ± 0.042** |

The P1 pre-solve took 164 ms here, against 7.8 s for one full sweep.

### 2.5 Q3: sparse stencil for the pirate expectation (disagree)

Benchmarked with Gemini's exact code, on a single core, best of 2 runs:

| Pirate region size | States | Current einsum | Gemini stencil | Per-direction stencil (my variant) |
|---|---|---|---|---|
| 4 | 0.06M | **0.001 s** | 0.003 s | 0.002 s |
| 10 | 0.36M | **0.008 s** | 0.018 s | 0.011 s |
| 30 | 2.9M | **0.116 s** | 0.238 s | 0.140 s |
| 100 | 31.6M | 3.09 s | 5.97 s | **2.92 s** |

- All three give identical results (difference ≤ 3e-16).
- The O(5|R|) vs O(|R|²) flop argument doesn't carry over to practice. Fancy-index gathers on the innermost axis are memory-bound.
- Gemini's version also builds two arrays 5× the size of V, about 1.3 GB each on the blocks layout.
- Its claim that this would "dramatically cut the 7-second sweep" is false. Even an infinitely fast pirate stage would remove only about 40% of that sweep, because the ship stage, landing and memory traffic make up the rest.
- The TAs say real tests use small pirate regions, where einsum is fastest anyway.

### 2.6 Q4: the 2N² step cap (agree with the conclusion, not the reasoning)

Gemini argues that γ^1800 ≈ 1e-8, so the cap is negligible. That only holds for N=30:

| Test | N | Cap H = 2N² | γ^H |
|---|---|---|---|
| 3 | 5 | 50 | 0.605 |
| 1 | 10 | 200 | 0.134 |
| 2 | 30 | 1800 | 1.4e-8 |

So the discount argument fails for small grids. Instead I computed the exact answer.
- Synchronous value iteration from V=0, run for exactly H sweeps, *is* the optimal H-step (time-aware) value V*_H.
- Running H fixed-policy backups gives our stationary policy's H-step value V^π_H.

| Test | V^π_H(start), our policy under the cap | V*_H(start), best possible under the cap | Maximum possible gain |
|---|---|---|---|
| 3 | 7.1485 | 7.1485 | 9e-10 |
| 1 | 6.0678 | 6.0678 | 0 |
| 2 | 1.7466 | 1.7466 | 0 |

The real reason the cap doesn't matter is that our episodes end long before it (mean 16, 44 and 87 steps against caps of 50, 200 and 1800).

Gemini's feasibility argument is also wrong. Backward induction needs H sweeps, not a state space multiplied by 1800, and we already run 55–193 sweeps. Only the per-step policy would need storing. The real obstacle is that `get_action` doesn't receive a time step: we would have to infer it by counting calls and detecting episode starts, which is fragile.

**No change.**

---

## 3. Part B plan: details

| Item | Finding |
|---|---|
| 5a. Table size | 4·4·5⁴ = **10,000 states**, × 5 actions = 50,000 entries (`np.zeros((4,4,5,5,5,5,5)).size == 50000`). Gemini's "100,000 states" is an arithmetic slip. Harmless. |
| 5b. Optimistic init 5.0 | **TA rule (Piazza):** "You cannot assume any information other than state representation from the environment including any kind of reward function." Gemini's justification (the −5 collision reward) relies on exactly that. (continued below) |
| 5c. ε decay by elapsed time | Agree. Already in our plan (B4). |
| 5d. Fresh `HighwayEnv()` per episode | Agree. Already in our plan (B2). Measured at 0.25 ms per construction, about 33 µs per step overall, so it is negligible. |

More on 5b:
- **Scale.** The per-step reward is 0.03·speed, between 0.03 and 0.12 (`Car.step` returns speed·0.3, times 0.1). So the largest possible return is about 12 at γ=0.99, and 5.0 is *not* optimistic for fast states. At γ=0.9 (and the TAs say γ may change) the largest return is about 1.2, so 5.0 would be 4× too optimistic and slow to wash out.
- **Gemini's reasoning is backwards.** It says zero initialization makes the agent "fearful of unexplored states". In fact every reward except a collision is positive, so with zero initialization an unvisited action looks *worse* than a visited safe one. Zero initialization is therefore already pessimistic, which reduces exploration.
- **So some optimism might help.** But it should come from data (for example, the largest per-step reward observed so far divided by (1−γ), using the γ given to the agent), and it should be compared against zero initialization in experiments, not assumed.

---

## 4. Proposed changes (for discussion, not implemented)

### P1: relaxed pirate-free pre-solve to initialize V and the policy (recommend adopting)

**What**
1. At the start of `learn_policy`, solve the small MDP over (mask, ship cell). Use the exact wind, fort, treasure and step rules, but leave out the pirates. That is 4×K states, so it takes milliseconds.
2. Set `V[m, s, r1, r2] = V_relaxed[m, s]` for every pirate configuration.
3. Set the initial policy to the relaxed greedy action, broadcast across pirate configurations. This is free and needs no full sweep.
4. Then run the existing exact factored sweeps unchanged.

**Why**
- In every measurement above it gives the best anytime quality and 1.1–1.9× faster convergence.
- On the blocks layout at T=60 the score goes from −5.0 to +1.9.
- The final answer is unchanged, because value iteration converges to V* from any starting point (verified: final error ≤ 7e-9 on all layouts).

**Cost**
- 2–35 ms on realistic layouts and about 160 ms on the 31.5M-state layout.
- The pre-solve can use a looser tolerance (for example 1e-6), and it should also check the deadline.

**Caveats**
- The relaxed values ignore pirates, so early on they are too optimistic near pirate lanes, and the earliest policy may cross lanes carelessly. The full sweeps then correct this. In all 4 layouts the early policy was already near-optimal, but a layout dominated by pirate risk might show a weaker early policy. It would still be far better than the current all-UP / −5 behaviour.
- V_relaxed is not a guaranteed upper or lower bound on V*, because ramming a pirate can beat wandering. This doesn't affect correctness.

**Possible extension, P1b (not measured; only if P1 shows problems):** include pirate risk in the relaxation, using each pirate's stationary occupancy probability.

### P2: catch `run.py`'s `TimeoutException` inside `learn_policy` (already planned in A5)

Gemini did not comment on this. Together with P1 it guarantees a sensible policy even if the very first full sweep overruns a tiny budget. Today the policy before the first sweep is all UP.

### Roadmap edits (proposed)

- **A5:** replace "solve the treasure layers in order (mask 0 first …)" with **P1**. Keep the per-sweep clock guard and P2. Drop layered solving, Gemini's BFS initialization, Gauss–Seidel and the stencil, for the reasons in section 2.
- **B4:** change "try optimistic initial Q values" to "try optimism derived from observed data only (no reward constants), compared against zero initialization".

---

## 5. Reproducing the measurements

- `tools/experiments/proto_variants.py`: prototypes of all variants, built on top of the current `Agent` without modifying it.
- `tools/experiments/compare_variants.py <layout> <prob> zeros,layered,bfs_gemini,relaxed,layered+relaxed,gs,gs+relaxed`: prints the tables in 2.4.
- `tools/experiments/layouts/`: the synthetic 30×30 layouts (small, corridors, blocks, maze).
- `tools/brute_check_a.py`, `tools/fuzz_a.py`, `tools/eval_a.py`: correctness checks and the evaluation harness from steps A1–A4.
