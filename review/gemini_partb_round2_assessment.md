# Assessment of Gemini's Part B review (round 2)

**Status: `part_b/agent.py` has not been changed.**

**How this was verified:**
- **Experiments:** 115 runs, single core per run, 300 greedy evaluation episodes each, with settings interleaved in one job queue. Raw lines are in `tools/experiments/part_b/results/verify_round2_results.txt`.
- **Independent review:** a 7-agent workflow (math audit, mechanism analysis, signal safety, code review, plus adversarial verification of each code-review finding). Raw output is in `.../verify_round2_review_agents.jsonl`.
- **Pre-registration:** the mechanism agent wrote its predictions while the queue was still on the T=240 runs (28/115 done). Every mechanism run was a T=60 run queued after them, so no mechanism result existed when it predicted.
- **Timeout stress test:** `tools/experiments/part_b/timeout_stress.py`.

---

## 1. Summary

| # | Gemini's claim | Verdict | Key evidence |
|---|---|---|---|
| 1 | The aggressive optimum is real; "crashing late at high speed strictly dominates surviving at low speed"; the "undiscounted return will likely be lower or volatile" | **Conclusion correct; two sub-claims wrong** | Arithmetic exact (11.41, −0.245, 3.0). "Strictly dominates" holds only for crashes after K* = 64/85/130 steps (speed 4/3/2, γ=0.99). Measured: crashes cost only −0.35 to −0.48 of discounted score, median crash step 354–413. The undiscounted return is **higher** (47–48 vs 41–45), not lower. |
| 2 | A larger α helps because of non-stationarity; a small α "computes an arithmetic mean over millions of steps" | **Wrong** | A constant α=0.1 is an EMA with a memory of about 9 updates, not an arithmetic mean. The decisive experiment: **α=0.1 with Q₀=+3 scores 6.52 against 3.93 with Q₀=0**, with the same α and the same ε decay. The bottleneck is **pessimistic initial values propagating slowly**, not lag behind changing behaviour. |
| 3 | γ=0.5 → "the penalty vanishes, more reckless"; γ=0.999 → crash at step 300 costs −3.7; "the α schedule should transfer cleanly"; may need γ-dependent tuning | **Mixed. The schedule does transfer, measured, but not for Gemini's reasons** | Tuned ≥ α=0.3 ≥ α=0.1 at every γ ∈ {0.5…0.999}, at T=60 and T=240. At γ=0.5 a crash on the next step costs 2.5, about 14× the whole speed upside; "reckless" is optimal because no crash can happen before step 7. At γ=0.999 the real cost of a crash is the lost continuation (about 48), not −3.7, and the agent indeed becomes cautious (7% crashes). |
| 4 | "Training until 0.95·T is safe; interrupting is harmless; the handler keeps Q consistent" | **Partly correct** | Safe *with* a safety net: 300 randomly timed interruptions, 0 escaped exceptions, 0 invalid Q values, and the real `run.py` with a forced alarm still scored. But (a) **the current Part B agent has no safety net**, so "harmless" is false today; and (b) **0.95·T gave no gain** (T=60: 5.46 vs 5.61; T=240: 6.87 vs 7.10). Stopping later also lowers the final ε (0.16 → 0.06). |
| 5 | Code review: list Q-table "accounts for the 6%"; step-based α correct; done handling correct; zero-tie default "optimal"; "the implementation is optimal" | **Mostly right on mechanics; "optimal" is wrong** | Hardening gaps: no timeout net, and the deadline uses `time.time()` instead of a monotonic clock. Nits: a stale comment, an unreachable `alpha_decay_frac` branch, `eps_end` never reached (ε ends at about 0.16). The 6% is our own measurement, not Gemini's. |

---

## 2. Claim 1: the aggressive optimum (measured)

**Per-episode decomposition** (`episode_stats.py`, T=240, γ=0.99, 300 episodes per run):

| Agent | Discounted score | Driving reward | Crash penalty | Undiscounted | Crash rate | Crash step p10 / p50 / p90 |
|---|---|---|---|---|---|---|
| Tuned (run 1) | 7.45 | 7.80 | **−0.35** | **47.2** | 66% | 94 / 413 / 815 |
| Tuned (run 2) | 7.01 | 7.50 | **−0.48** | **48.4** | 59% | 49 / 354 / 854 |
| α=0.1 (run 1) | 4.49 | 4.59 | −0.09 | 40.7 | 11% | 86 / 245 / 806 |
| α=0.1 (run 2) | 4.84 | 4.93 | −0.09 | 44.8 | 12% | 59 / 399 / 783 |

- **Most crashes come late.** The median is well past the break-even crash step, K* = 64 / 85 / 130 for speed 4 / 3 / 2 at γ=0.99. So the speed gained (+3.0 in driving reward) is worth far more than the penalty paid (about −0.3).
- **Gemini's "undiscounted will likely be lower" is wrong.** The undiscounted return is higher (47–48 vs 41–45). For Gemini's own example, 300 steps at speed 4 and then a crash gives 31.0, against 30.0 for speed 1 over 1000 steps.
- **"Strictly dominates" is too strong.** It depends on crash timing. A crash at step 30 at speed 4 scores −0.58, far below the safe 3.0. About 10% of the tuned agent's crashes come before step 49–94, near or below break-even, which is room for improvement.

**Break-even crash step K\*** (math audit; smallest K at which "speed s for K steps, then crash" beats crash-free speed 1 over the 1000-step cap):

| γ | 0.5 | 0.8 | 0.9 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|
| speed 4 / 3 / 2 | 5 / 6 / 7 | 12 / 13 / 16 | 19 / 22 / 28 | 28 / 34 / 46 | 64 / 85 / 130 | 213 / 291 / 460 |

---

## 3. Claim 2: why a larger α helps (decisive experiments)

All at T=60, γ=0.99, 3 runs each (diagnostic variants run from a scratch copy; Q₀=+3 uses reward knowledge and is **not** allowed in the submission).

| Variant | Score | Crash | Mean speed index | Pre-registered prediction | Matched? |
|---|---|---|---|---|---|
| M1 constant α 0.1 (ε decays) | 3.93 | 19% | 0.29 | 3.8 | ✓ |
| **M2 α 0.1, Q₀ = +3** | **6.52** | 90% | 1.86 | 4.6 (4.2–5.2) | direction ✓, size far bigger |
| M3 α = max(0.1, 1/n(s,a)) | 4.26 | 38% | 0.59 | 3.95 (3.6–4.3) | ✓ |
| M7 α = max(0.25, 1/n(s,a)) | 4.30 | 14% | 0.40 | 4.35 | ✓ |
| **M4 α 0.1, constant ε 0.3** | **5.68** | 100% | 2.86 | 3.6 ("no improvement") | ✗ |
| M5 α 0.5, constant ε 0.3 | 4.26 | 100% | 1.99 | 5.1 | ✗ |
| M6 tuned schedule (adopted) | 5.32 | 71% | 1.10 | 5.35 | ✓ |

**Interpretation:**
- **M2 is decisive.** It keeps α=0.1 *and* the same ε decay, so any "lag behind changing behaviour" is unchanged. Changing only Q₀ from 0 to +3 lifts the score from 3.93 to **6.52**, above even the tuned agent. The limiting factor is that Q starts at 0 while true values are about 3–12 (every non-collision reward is positive):
  - With bootstrapping, that bias decays only by a factor of (1 − α(1−γ)) per update, which is about 1000 updates at α=0.1.
  - The −5 crash target is non-bootstrapped and is learned in about 10 updates.
  - So rarely visited fast states look worse than they are, and the agent stays slow.
  - A larger α speeds up that propagation, which is why it helped.
- **M4 points the same way.** Constant ε=0.3 (no decay, so no behaviour-driven non-stationarity) gives many more updates in fast states and lifts α=0.1 to 5.68.
- **M5 shows the interaction.** With heavy exploration, a large α only adds noise (4.26). α and exploration have to be tuned together.
- **Gemini's specific description is wrong:** "arithmetic mean over millions of steps" describes α=1/n, not a constant α. Q-learning's `max` target also doesn't depend on the behaviour ε directly.

**Important consequence for B4:** two levers bigger than the α schedule exist.
- **Optimism.** M2: +1.2 over the tuned agent, but Q₀=+3 is reward knowledge. A TA-compliant version would have to come from observed data (for example, initialising from returns actually seen during training).
- **More exploration.** M4: +0.4 over the tuned agent. This uses no reward knowledge at all. Note that earlier B4(a) floors of 0.05 and 0.1 did not help, while a constant ε of 0.3 does, so the *level* matters.

---

## 4. Claim 3: γ robustness (measured)

Mean score (T=60: 3 runs; T=240: 2 runs):

| γ | T=60 tuned | T=60 α 0.3 | T=60 α 0.1 | T=240 tuned | T=240 α 0.1 | Crash rate (tuned) |
|---|---|---|---|---|---|---|
| 0.5 | 0.196 | 0.196 | 0.194 | 0.197 | 0.197 | 99–100% |
| 0.8 | **0.521** | 0.504 | 0.448 | 0.535 | 0.530 | 99–100% |
| 0.9 | 0.896 | 0.895 | 0.836 | **1.051** | 1.014 | 94–100% |
| 0.95 | 1.591 | 1.585 | 1.560 | 1.816 | 1.801 | 93–100% |
| 0.99 | **5.418** | 4.379 | 3.884 | **7.109** | 4.877 | 72–74% |
| 0.999 | **22.28** | 21.79 | 19.25 | **28.55** | 21.33 | 7% |

- **The tuned schedule is never worse,** and it is much better at γ ≥ 0.99. No γ-dependent tuning is needed on this evidence.
- **Behaviour adapts to γ by itself.** At γ ≤ 0.95 every variant drives fast and almost always crashes. That is optimal there, because the score is settled within the first few steps and no crash is physically possible before step 7. At γ=0.999 the agent becomes cautious (7% crashes), as the math predicts.
- **For scale at γ=0.999:** crash-free speed 1 = 18.97 and crash-free speed 4 = 75.9. The tuned 22–29 beats safe-slow, but there is clearly headroom.

---

## 5. Claim 4: B5 time safety

- **Stress test** (prototype with a name-matched `TimeoutException` net around the whole training loop): 150 agents × 2 random-time interruptions → 0 escaped exceptions, 0 invalid Q/visits/actions. `learn_policy` returned within 0.06 ms (median), 2.5 ms (max) of the alarm. The real `run.py` with the alarm forced to fire still printed a score.
- **What Gemini missed:**
  - The **current** Part B agent has no net, so an alarm during training would score `None`.
  - Precisely, the handler runs only at eval-breaker checkpoints (function entry, loop back-edges, after calls). An interrupt can drop the one in-flight update. That is harmless: every Q slot is always a valid float.
  - Use a **monotonic** clock (`time.monotonic`) for the deadline. `time.time()` can jump, while SIGALRM can't.
- **0.95·T vs 0.85·T** (with the net, 3 runs each):

  | Stop at | T=60 | T=240 |
  |---|---|---|
  | 0.85·T | 5.61 | 7.10 |
  | 0.95·T | 5.46 | 6.87 |

  There is no gain despite 2–12% more steps, consistent with the final ε dropping from 0.16 to 0.06.

---

## 6. Claim 5: code review (independent agents, adversarially verified)

| Finding | Severity | After adversarial check |
|---|---|---|
| No `TimeoutException` safety net; deadline on `time.time()` | risk (hardening) | **real**: low probability, total cost (score `None`) |
| 1000-step limit treated like a collision (no bootstrap) | claimed risk | **not an issue**: `run.py` also stops scoring at step 1000, so `r_t + γ·r_{t+1}` is the exact scored return from there |
| Hyperparameters validated only at γ=0.99 | claimed risk | **not a code defect**; now measured (section 4): fine |
| Unvisited states default to action 0 | risk | measured earlier: 0.01–0.1% of steps, no effect in an A/B test |
| Stale comment "refined in step B5"; unreachable `alpha_decay_frac` branch; `eps_end` comment inaccurate (ε ends at about 0.16); repeated `learn_policy` calls restart ε (irrelevant for `run.py`) | nit | cleanup |

---

## 7. Proposals (for discussion, nothing implemented)

1. **B5 (hardening, recommended):**
   - Add the name-matched `TimeoutException` net around the whole training body, returning immediately.
   - Use `time.monotonic()`.
   - **Keep the 0.85·T stop.** 0.95·T measured no better.
   - Clean up the nits.
2. **B4 round 2: the two new levers.** Each is to be measured at T=60 and T=240 and across γ before adopting anything.
   - **(a) Exploration level, no reward knowledge needed.** Sweep a constant or slowly decaying ε ∈ {0.2, 0.3, 0.5} with small α (0.1–0.2). M4 already beats the tuned agent at T=60 (5.68 vs 5.32).
   - **(b) Data-driven optimism, TA-compliant.** Replace Q₀=0 with an optimistic value learned from experience, for example the discounted return of early episodes or the running maximum of observed per-step reward / (1−γ). It must never be a hard-coded reward constant. M2 shows the upside (6.52 vs 5.32), but only a TA-compliant variant may ship.
3. **Report:** document the finding that the pessimistic Q₀ plus slow bootstrapped propagation is the real bottleneck. It explains why a large α, more exploration and optimism all help.
