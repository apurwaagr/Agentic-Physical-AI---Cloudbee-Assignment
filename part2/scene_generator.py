# scene_generator.py  -  Task 2A
# Two functions are intentionally wrong (marked STUB).
# Fix both, then run: pytest test_2a.py  -  all 3 tests must pass.
#
# After passing all tests:
#   python scene_generator.py                -> generates 100 halton + 100 random configs
#   Plot obj_x vs obj_y AND friction vs light_int (both strategies side by side)
#   Save the plot as plots/coverage.png
#   Add one new parameter of your choice to PARAM_SPACE with a comment justifying it

import numpy as np
import json
import pathlib


PARAM_SPACE = {
    "obj_x":     (-0.3,  0.3),   # object x position (metres)
    "obj_y":     (-0.3,  0.3),   # object y position (metres)
    "obj_z":     ( 0.02, 0.15),  # object height (metres)
    "obj_scale": ( 0.5,  2.0),   # scale factor
    "friction":  ( 0.1,  1.0),   # surface friction coefficient
    "light_int": ( 0.3,  1.0),   # lighting intensity
    "cam_yaw":   (-30,   30),    # camera yaw (degrees)
    "obj_mass":  ( 0.05, 1.0),   # object mass (kg) — added because mass directly affects grasp dynamics and contact forces, making it a critical sim-to-real gap parameter
}

KEYS = list(PARAM_SPACE.keys())


def _halton(index: int, base: int) -> float:
    """Return one value of the Halton low-discrepancy sequence."""
    result = 0.0
    denominator = 1.0
    n = index
    while n > 0:
        denominator *= base
        n, remainder = divmod(n, base)
        result += remainder / denominator
    return result


def validate_config(cfg: dict) -> bool:
    """Return True only if cfg contains all required keys and every value
    lies within [lo, hi] for its corresponding PARAM_SPACE entry."""
    if not all(k in cfg for k in PARAM_SPACE):
        return False
    for key, (lo, hi) in PARAM_SPACE.items():
        if not (lo <= cfg[key] <= hi):
            return False
    return True


def sample_random(n):
    """Sample n configurations using uniform random sampling."""
    lo = np.array([v[0] for v in PARAM_SPACE.values()])
    hi = np.array([v[1] for v in PARAM_SPACE.values()])
    return [dict(zip(KEYS, lo + np.random.rand(len(KEYS)) * (hi - lo)))
            for _ in range(n)]


def sample_halton(n):
    """Sample n configurations using Halton low-discrepancy sequence."""
    bases = [2, 3, 5, 7, 11, 13, 17, 19]  # 8 bases — one per parameter in PARAM_SPACE
    lo = np.array([v[0] for v in PARAM_SPACE.values()])
    hi = np.array([v[1] for v in PARAM_SPACE.values()])
    raw = np.array([[_halton(i + 1, b) for b in bases] for i in range(n)])
    return [dict(zip(KEYS, lo + raw[i] * (hi - lo))) for i in range(n)]


def generate(n=100, strategy="random", out_dir="configs/"):
    """Generate n scene configurations and write them as JSON files."""
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    configs = (sample_random if strategy == "random" else sample_halton)(n)
    valid = [c for c in configs if validate_config(c)]
    for i, c in enumerate(valid):
        json.dump(c, open(f"{out_dir}/scene_{i:04d}.json", "w"), indent=2)
    return valid


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pathlib.Path("plots").mkdir(exist_ok=True)

    halton_configs = generate(n=100, strategy="halton", out_dir="configs/halton")
    random_configs = generate(n=100, strategy="random", out_dir="configs/random")

    # Extract values for plotting
    hx = [c["obj_x"] for c in halton_configs]
    hy = [c["obj_y"] for c in halton_configs]
    rx = [c["obj_x"] for c in random_configs]
    ry = [c["obj_y"] for c in random_configs]

    hf = [c["friction"]  for c in halton_configs]
    hl = [c["light_int"] for c in halton_configs]
    rf = [c["friction"]  for c in random_configs]
    rl = [c["light_int"] for c in random_configs]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].scatter(hx, hy, s=10, alpha=0.7)
    axes[0, 0].set_title("Halton: obj_x vs obj_y")
    axes[0, 0].set_xlabel("obj_x"); axes[0, 0].set_ylabel("obj_y")

    axes[0, 1].scatter(rx, ry, s=10, alpha=0.7, color="orange")
    axes[0, 1].set_title("Random: obj_x vs obj_y")
    axes[0, 1].set_xlabel("obj_x"); axes[0, 1].set_ylabel("obj_y")

    axes[1, 0].scatter(hf, hl, s=10, alpha=0.7)
    axes[1, 0].set_title("Halton: friction vs light_int")
    axes[1, 0].set_xlabel("friction"); axes[1, 0].set_ylabel("light_int")

    axes[1, 1].scatter(rf, rl, s=10, alpha=0.7, color="orange")
    axes[1, 1].set_title("Random: friction vs light_int")
    axes[1, 1].set_xlabel("friction"); axes[1, 1].set_ylabel("light_int")

    plt.tight_layout()
    plt.savefig("plots/coverage.png", dpi=150)
    print("Saved plots/coverage.png")
    print(f"Halton: {len(halton_configs)} valid configs")
    print(f"Random: {len(random_configs)} valid configs")
