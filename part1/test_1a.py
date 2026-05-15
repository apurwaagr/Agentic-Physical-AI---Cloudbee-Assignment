# test_1a.py  -  run with: python test_1a.py
# All assertions must pass after you fix the 3 bugs in ddpm_scheduler.py

import torch
import math
from ddpm_scheduler import DDPMScheduler


def test_beta_schedule_direction():
    """Betas should increase from beta_start to beta_end (small to large)."""
    sched = DDPMScheduler(T=1000, beta_start=1e-4, beta_end=0.02)
    assert sched.betas[0] < sched.betas[-1], (
        f"Betas must be monotonically increasing: "
        f"betas[0]={sched.betas[0]:.6f}, betas[-1]={sched.betas[-1]:.6f}"
    )
    assert abs(sched.betas[0].item() - 1e-4) < 1e-6, "betas[0] must equal beta_start"
    assert abs(sched.betas[-1].item() - 0.02) < 1e-6, "betas[-1] must equal beta_end"


def test_alpha_bar_is_product_not_sum():
    """alpha_bar must be the CUMULATIVE PRODUCT of alphas, not the cumulative sum."""
    sched = DDPMScheduler(T=1000)
    # alpha_bar must lie in (0, 1] and decrease monotonically
    assert (sched.alpha_bar > 0).all(), "alpha_bar values must be positive"
    assert (sched.alpha_bar <= 1).all(), "alpha_bar values must be <= 1"
    assert sched.alpha_bar[0] > sched.alpha_bar[-1], (
        "alpha_bar must decrease over time (more noise at higher t)"
    )
    # Spot-check: manual cumprod for first 5 steps
    manual = torch.cumprod(1.0 - sched.betas[:5], dim=0)
    assert torch.allclose(sched.alpha_bar[:5], manual, atol=1e-6), (
        "alpha_bar[:5] does not match manual cumprod"
    )


def test_q_sample_signal_and_noise():
    """q_sample must scale x0 by sqrt_ab and noise by sqrt_1ab (not swapped)."""
    sched = DDPMScheduler(T=1000)
    torch.manual_seed(42)
    x0    = torch.ones(2, 1, 4, 4)   # constant signal
    noise = torch.zeros(2, 1, 4, 4)  # zero noise
    t     = torch.tensor([0, 0])

    # With zero noise, output must equal sqrt_ab[0] * x0
    out = sched.q_sample(x0, t, noise)
    expected = sched.sqrt_ab[0] * x0
    assert torch.allclose(out, expected, atol=1e-6), (
        f"With zero noise, q_sample must return sqrt_ab * x0. "
        f"Got mean={out.mean():.4f}, expected mean={expected.mean():.4f}"
    )

    # At t~999 (heavy noise), the noisy sample must differ from x0
    noise2 = torch.randn(2, 1, 4, 4)
    t999   = torch.tensor([999, 999])
    out2   = sched.q_sample(x0, t999, noise2)
    assert not torch.allclose(out2, x0, atol=0.1), (
        "At t=999 q_sample should differ substantially from x0"
    )


def test_snr_decreasing():
    """SNR must be high at t=0 (clean) and near zero at t=999 (noisy)."""
    sched = DDPMScheduler(T=1000)
    snr_0   = sched.snr(0)
    snr_500 = sched.snr(500)
    snr_999 = sched.snr(999)
    assert snr_0 > snr_500 > snr_999, (
        f"SNR must strictly decrease: snr(0)={snr_0:.2f}, "
        f"snr(500)={snr_500:.4f}, snr(999)={snr_999:.6f}"
    )
    assert snr_999 < 0.01, f"SNR at t=999 should be near zero, got {snr_999:.6f}"


if __name__ == "__main__":
    tests = [
        test_beta_schedule_direction,
        test_alpha_bar_is_product_not_sum,
        test_q_sample_signal_and_noise,
        test_snr_decreasing,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed < len(tests):
        raise SystemExit(1)
