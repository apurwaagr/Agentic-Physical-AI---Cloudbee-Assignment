# Part 1B — Video Diffusion Written Answers

## Q1: How does the DDPM forward process differ in video diffusion vs image diffusion?

In standard image DDPM, each frame is treated independently — noise is added to a single 2D image over T timesteps. Video diffusion models extend this to a sequence of frames (3D tensor: T×H×W×C) and must preserve temporal coherence. This means the noising schedule is applied jointly across frames, and the neural network (typically a 3D U-Net or a Transformer) uses **spatio-temporal attention** to model correlations both within each frame and across time. Without cross-frame attention, consecutive predicted frames would be temporally inconsistent (flickering, discontinuous motion).

## Q2: What is classifier-free guidance (CFG) and why is it used?

Classifier-free guidance (Ho & Salimans, 2021) allows a single diffusion model to perform conditional generation without a separately trained classifier. During training, the conditioning signal (e.g. a text prompt or action label) is randomly dropped (replaced with null token ∅) with probability p_drop ≈ 0.1–0.15. At inference, two score estimates are computed — one conditioned (ε_θ(x_t | c)) and one unconditional (ε_θ(x_t | ∅)) — and combined:

```
ε_guided = ε_θ(x_t | ∅) + w · (ε_θ(x_t | c) − ε_θ(x_t | ∅))
```

The guidance scale `w > 1` amplifies the conditional direction, trading diversity for prompt fidelity. For robotic video prediction, CFG lets the model be conditioned on task instructions or goal images while retaining the ability to generate diverse rollouts.

## Q3: Why does action-conditioned video prediction matter for robotics?

A robot policy trained purely on observations cannot distinguish which future outcomes result from different actions. Action-conditioned video prediction generates plausible future frames **given a specific action sequence**, enabling:

1. **Model-based planning**: Roll out candidate action sequences in imagination and select the one that reaches the goal state.
2. **Data augmentation**: Synthesise robot demonstrations for states not covered in the real training set, directly addressing the data scarcity problem.
3. **Reward shaping**: Use the predicted video to estimate whether a goal is achievable before committing to a real-world trajectory.

Key mechanism: the video diffusion U-Net receives the proposed action as an additional conditioning input (typically concatenated to the noise at each denoising step or injected via cross-attention), so the denoised sequence is causally linked to the action rather than being unconditional imagination.
