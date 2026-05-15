# app.py  —  Task 3B: FastAPI scene generation endpoint
# ─────────────────────────────────────────────────────────────────────────────
# WHAT YOU MUST IMPLEMENT:
#   - POST /generate endpoint
#   - Import and call scene_generator.py from part2/
#   - Compute a coverage_score (your own metric is fine — justify it briefly)
#   - Return the response schema exactly as specified
#
# RUN locally:   uvicorn app:app --reload --port 8000
# RUN via Docker (after completing Dockerfile):
#   docker build -t scene-gen .
#   docker run -p 8000:8000 scene-gen
# ─────────────────────────────────────────────────────────────────────────────

import sys
import pathlib
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add part2 to path so we can import scene_generator
# Adjust this path if your directory structure differs
sys.path.append(str(pathlib.Path(__file__).parent.parent / "part2"))

app = FastAPI(title="Scene Generator API", version="1.0.0")


# ── Request / Response models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    task:     str    # free-text description of the task (logged but not used)
    n:        int    # number of scenes to generate
    strategy: str    # "random" or "halton"

class GenerateResponse(BaseModel):
    configs:        List[dict]   # list of valid scene configuration dicts
    n_valid:        int          # number of configs that passed validate_config
    coverage_score: float        # scalar in [0, 1] — your coverage metric


# ── Coverage score helper ─────────────────────────────────────────────────────

def compute_coverage_score(configs: list) -> float:
    """
    Coverage score based on mean nearest-neighbour distance, normalised to [0, 1].

    Intuition: if every point is far from its nearest neighbour, the space is well-covered.
    We compute the mean NN distance across all configs, then divide by the theoretical
    maximum (diagonal of the unit hypercube in `d` dimensions = sqrt(d)), giving a value
    in [0, 1] where 1 means perfectly spread out.

    This is a standard space-filling quality metric used in design-of-experiments.
    """
    if not configs or len(configs) < 2:
        return 0.0

    import numpy as np
    from scene_generator import PARAM_SPACE

    keys = list(PARAM_SPACE.keys())
    lo   = np.array([PARAM_SPACE[k][0] for k in keys], dtype=float)
    hi   = np.array([PARAM_SPACE[k][1] for k in keys], dtype=float)
    span = hi - lo
    span[span == 0] = 1.0  # avoid division by zero

    # Normalise each config to [0,1]^d
    pts = np.array([[( c[k] - lo[i]) / span[i] for i, k in enumerate(keys)]
                    for c in configs])

    # Nearest-neighbour distances
    nn_dists = []
    for i, p in enumerate(pts):
        diffs = pts - p
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists[i] = np.inf   # exclude self
        nn_dists.append(dists.min())

    mean_nn = float(np.mean(nn_dists))
    max_possible = float(np.sqrt(len(keys)))   # diagonal of unit hypercube
    return min(mean_nn / max_possible, 1.0)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse)
def generate_scenes(request: GenerateRequest):
    """
    Generate N scene configurations using the requested sampling strategy.
    Returns only the valid configs (those passing validate_config).
    """
    if request.strategy not in ("random", "halton"):
        raise HTTPException(status_code=400,
            detail="strategy must be 'random' or 'halton'")
    if request.n < 1 or request.n > 10_000:
        raise HTTPException(status_code=400,
            detail="n must be between 1 and 10000")

    # Import and call scene_generator.generate()
    from scene_generator import generate
    configs = generate(n=request.n, strategy=request.strategy, out_dir="/tmp/configs")

    score = compute_coverage_score(configs)

    return GenerateResponse(
        configs        = configs,
        n_valid        = len(configs),
        coverage_score = score,
    )


@app.get("/")
def root():
    """Root endpoint with API documentation."""
    return {
        "title": "Scene Generator API",
        "version": "1.0.0",
        "description": "Generate robotic task scene configurations using Halton or random sampling",
        "endpoints": {
            "POST /generate": {
                "description": "Generate scene configurations",
                "request_body": {
                    "task": "str (task description, e.g., 'pick-and-place')",
                    "n": "int (number of scenes to generate, 1-10000)",
                    "strategy": "str ('random' or 'halton')"
                },
                "example": "curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' -d '{\"task\":\"pick-and-place\",\"n\":10,\"strategy\":\"halton\"}'"
            },
            "GET /health": "Health check endpoint"
        },
        "note": "This API requires POST requests. Use curl or a REST client to test."
    }


@app.get("/health")
def health():
    return {"status": "ok"}
