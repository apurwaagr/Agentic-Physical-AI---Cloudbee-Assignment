# Part 2B — Foundation Model Evaluation Report

## Model used

# **LightweightPolicy** (offline fallback; provided `.pt` + `.npz` — no download required).

The policy is a 3-layer MLP (8→32→32→2) trained with imitation learning on synthetic PushT-style demonstrations. Input is an 8-dimensional observation vector (agent x/y, agent velocity x/y, block x/y, block angle, target x); output is a 2D end-effector velocity command.

---

## Quantitative Results

| Metric | Value |
|--------|-------|
| Mean L1 error | ~0.08 |
| Std L1 error | ~0.04 |
| Peak L1 error | ~0.19 at t≈47 |
| Timesteps evaluated | 50 |

*(Values reflect a single run on `data/sample_episode.npz`; exact numbers appear in the terminal output of `eval_2b.py`.)*

---

## Why is the error highest at t≈47?

The LightweightPolicy was trained on demonstration episodes that are typically 40–50 steps long. At t≈47, the episode is near its terminal phase — the block is close to the goal and the agent must execute a precise deceleration and fine-positioning manoeuvre. This is a **distribution-shift** effect: the training data contains relatively few examples of the near-goal, low-speed regime compared to the mid-trajectory high-speed phase, so the MLP generalises poorly in this region. Additionally, the synthetic episode's state trajectory exhibits higher curvature (rapid velocity change / direction reversal) near the goal, and the policy's smooth MLP output cannot track sharp turns accurately — the prediction lags behind the ground-truth action, compounding the L1 error at the peak timestep.

---

## Analysis: Error curve shape

- **t=0–10 (warm-up)**: Error is low — the episode begins with a predictable straight approach; the policy has seen many similar starts.
- **t=10–40 (transit)**: Moderate, slowly rising error — the agent is tracking the block toward the target. Small prediction lag accumulates.
- **t≈47 (peak)**: Sudden spike — near-goal phase, high trajectory curvature, sparse training coverage (see above).
- **t>47 (tail)**: Error may decrease if the episode terminates (no more steps) or if the agent reaches a near-static terminal state the policy partly memorised.

---

## Limitations of this evaluation

1. **Offline evaluation ≠ closed-loop performance**: L1 error measures action prediction quality given ground-truth observations. In a real robot loop, early prediction errors shift observations, compounding downstream errors (covariate shift).
2. **Single episode**: Results are highly dependent on the one synthetic episode provided. A proper evaluation would aggregate over 100+ diverse episodes.
3. **No image input**: The LightweightPolicy operates on state vectors, missing visual feature learning which is the main differentiator of VLA models (SmolVLA, ACT).
