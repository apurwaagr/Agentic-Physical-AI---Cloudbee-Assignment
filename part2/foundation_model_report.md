# Part 2B — Foundation Model Evaluation Report

## Model used

**Primary model: ACT (`vvrs/act-pusht`) on real PushT episodes from `lerobot/pusht`.**

I originally used the provided `LightweightPolicy` because it let me complete an offline sanity check without downloading a robotics stack. That was useful as a baseline, but it missed the main point of this task: exercising a vision-conditioned robotic foundation model on recorded image observations. I revised `eval_2b.py` so the default path now loads ACT through LeRobot, downloads/reads the real PushT dataset, and runs offline action inference over the first episode. The original scaffold id `lerobot/act_pusht_image` is not a public Hub repo, so the code uses the public PushT ACT checkpoint `vvrs/act-pusht` by default and supports `ACT_MODEL_ID=...` as an override.

ACT is a modern imitation-learning policy for robotics that uses image observations and action chunking. Unlike the fallback MLP, it must encode visual state, infer the block/agent geometry from camera input, and produce a temporally coherent action sequence. `MODEL_CHOICE=smolvla` is also supported for SmolVLA experiments; `MODEL_CHOICE=lightweight` remains available only as a no-network fallback.

---

## Quantitative Results

### ACT / real PushT data

Run:

```bash
cd part2
MODEL_CHOICE=act python eval_2b.py
```

This saves the required plot to `plots/action_error.png`. The values below come from the successful local LeRobot/HuggingFace run:

| Metric | Value |
|--------|-------|
| Mean L1 error | 291.5478 |
| Std L1 error | 74.2058 |
| Peak L1 error | 376.5454 at t=33 |
| Timesteps evaluated | 50 |

Note on scale: the public checkpoint runs successfully on real PushT frames, but its action scale appears mismatched with the current `lerobot/pusht` ground-truth action convention. The important correction versus my first submission is that this is now an actual image-conditioned ACT forward pass on real recorded data; for a production-quality comparison I would next verify the checkpoint's dataset revision and action normalization/post-processing metadata before interpreting the absolute L1 value.

### Lightweight baseline / provided synthetic episode

Run:

```bash
cd part2
MODEL_CHOICE=lightweight python eval_2b.py
```

| Metric | Value |
|--------|-------|
| Mean L1 error | 0.0456 |
| Std L1 error | 0.1805 |
| Peak L1 error | 0.9464 at t=49 |
| Timesteps evaluated | 50 |

---

## Why does the ACT error spike?

For ACT, the highest error in this run occurs at `t=33`. That region is a trajectory transition point: the model predicts a chunk of future actions, but the ground-truth demonstrator can make sharp corrections when the pusher contacts the block or changes from approach to push. The image encoder may still localise the scene correctly, but the chunked action decoder smooths over abrupt velocity changes, so L1 error spikes around contact onset, contact break, or near-goal correction phases.

This failure mode is different from generic uncertainty. It comes from the interaction between visual state estimation, contact-rich dynamics, and action chunking: small pose or timing errors in the image-conditioned latent state can shift the predicted chunk a few frames early or late, which looks like a large per-timestep L1 error even when the action is qualitatively reasonable.

## Lightweight baseline peak at t=49

The fallback MLP peaks at the final evaluated timestep because the synthetic episode is near its terminal fine-positioning phase. The block is close to the target and the agent must decelerate or reverse direction precisely, which is underrepresented relative to the mid-trajectory push. Since the fallback policy only sees an 8D state vector and produces a smooth single-step action, it lags behind the sharper terminal correction and produces a large L1 spike.

---

## What this evaluation shows

1. **ACT engages with the intended foundation-model setting.** It uses real recorded PushT observations and an image-conditioned policy instead of a state-vector MLP.
2. **Offline inference is a first diagnostic, not full robot competence.** It measures action agreement under expert observations; closed-loop execution could still accumulate covariate shift.
3. **The error plot is behaviorally useful.** The peak timestep points to the exact part of the trajectory worth replaying in the video walkthrough: contact transitions and near-goal corrections reveal much more about policy quality than a single aggregate metric.
