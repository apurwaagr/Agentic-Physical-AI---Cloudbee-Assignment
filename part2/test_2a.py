# test_2a.py  -  run with: pytest test_2a.py
# All 3 tests must pass after fixing the two stubs in scene_generator.py.

import numpy as np
import pytest
from scene_generator import _halton, validate_config, sample_halton, PARAM_SPACE, KEYS


def test_halton_valid_range():
    """Halton values must lie in [0, 1) for any base and index >= 1."""
    for base in [2, 3, 5, 7, 11, 13, 17]:
        for idx in range(1, 200):
            val = _halton(idx, base)
            assert 0.0 <= val < 1.0, (
                f"_halton({idx}, {base}) = {val} is outside [0, 1)"
            )


def test_halton_low_discrepancy():
    """Halton sequence must distribute values more uniformly than the stub."""
    base = 2
    n = 64
    vals = [_halton(i + 1, base) for i in range(n)]

    # Split [0,1) into 8 buckets; each should receive at least 4 samples
    buckets = [0] * 8
    for v in vals:
        bucket = int(v * 8)
        buckets[min(bucket, 7)] += 1

    for i, count in enumerate(buckets):
        assert count >= 4, (
            f"Bucket {i} has only {count} samples — Halton is not distributing uniformly. "
            f"Counts: {buckets}"
        )


def test_validate_config_rejects_out_of_range():
    """validate_config must return False for configs with out-of-range values."""
    # Build a perfectly valid config
    good = {k: (lo + hi) / 2 for k, (lo, hi) in PARAM_SPACE.items()}
    assert validate_config(good) is True, "Valid config must return True"

    # Corrupt one value at a time
    for key, (lo, hi) in PARAM_SPACE.items():
        bad_low = dict(good)
        bad_low[key] = lo - 0.001
        assert validate_config(bad_low) is False, (
            f"Config with {key}={bad_low[key]:.4f} (below lo={lo}) must return False"
        )

        bad_high = dict(good)
        bad_high[key] = hi + 0.001
        assert validate_config(bad_high) is False, (
            f"Config with {key}={bad_high[key]:.4f} (above hi={hi}) must return False"
        )

    # Missing key must also fail
    missing_key = dict(good)
    del missing_key[KEYS[0]]
    assert validate_config(missing_key) is False, "Config missing a key must return False"
