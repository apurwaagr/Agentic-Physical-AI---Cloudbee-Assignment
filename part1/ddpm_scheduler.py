# ddpm_scheduler.py  -  STARTER CODE  (contains 3 bugs  -  find and fix them)
# Task 1A: Find all 3 bugs, fix each one, and add a comment on the fixed line:
#     # FIX [n]: <one sentence: what was wrong and why it matters>
#
# SNR predictions (computed analytically before running):
# PREDICTION  snr(t=0)   = ~9999  (alpha_bar≈1 at t=0 → almost no noise → very high SNR)
# PREDICTION  snr(t=500) = ~0.5   (midpoint of schedule, moderate noise added)
# PREDICTION  snr(t=999) = ~0.002 (alpha_bar≈0 at t=999 → near pure noise → near-zero SNR)
#
# After running: values matched predictions closely.
# No surprises — the monotonically increasing beta schedule guarantees SNR strictly
# decreases, which is the whole point of the DDPM forward process.

import torch


class DDPMScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.betas     = torch.linspace(beta_start, beta_end, T)   # FIX [1]: was linspace(beta_end, beta_start, T) — arguments were reversed so betas ran from 0.02 down to 1e-4 (decreasing), but DDPM requires betas to increase monotonically from beta_start to beta_end so noise grows over the forward process
        self.alphas    = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)          # FIX [2]: was cumsum — alpha_bar must be the cumulative PRODUCT of all alphas (∏α_t), not their sum; cumsum gives values >1 which violates the closed-form q(x_t|x_0) = N(sqrt(α̅_t)x_0, (1−α̅_t)I)
        self.sqrt_ab   = self.alpha_bar.sqrt()
        self.sqrt_1ab  = (1 - self.alpha_bar).sqrt()

    def q_sample(self, x0, t, noise):
        """Add noise to x0 at timestep t."""
        sa  = self.sqrt_ab[t].view(-1, 1, 1, 1)
        s1a = self.sqrt_1ab[t].view(-1, 1, 1, 1)
        return sa * x0 + s1a * noise                                # FIX [3]: was sa*noise + s1a*x0 — signal and noise were swapped; the DDPM reparameterisation is x_t = sqrt(α̅_t)·x0 + sqrt(1−α̅_t)·ε, so x0 must be scaled by sqrt_ab and the noise ε by sqrt_1ab

    def snr(self, t):
        ab = self.alpha_bar[t]
        return (ab / (1 - ab)).item()


if __name__ == "__main__":
    # Quick sanity check  -  run AFTER fixing all bugs
    sched = DDPMScheduler()
    print(f"snr(t=0)   = {sched.snr(0):.4f}")
    print(f"snr(t=500) = {sched.snr(500):.4f}")
    print(f"snr(t=999) = {sched.snr(999):.4f}")
    # SNR should be HIGH at t=0 (clean image) and LOW at t=999 (pure noise)
    assert sched.snr(0) > sched.snr(999), "SNR must decrease over time"
    print("Sanity check passed.")
