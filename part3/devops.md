# Part 3B — DevOps Notes

## Three CI/CD tools recommended for a robotics inference service

### 1. GitHub Actions
Used for automated testing on every pull request. The pipeline runs:
- `pytest part1/ part2/` — unit tests for scheduler and scene generator
- `python -c "import json; json.load(open('part3/execution_log.json'))"` — JSON schema validation
- Docker build smoke test: `docker build -t scene-gen . && docker run --rm -p 8000:8000 -d scene-gen && sleep 5 && curl -f http://localhost:8000/health && docker stop $(docker ps -q)`

This catches import errors, broken stubs, and broken Docker images before they reach main.

### 2. Docker + GitHub Container Registry (GHCR)
The Dockerfile produces a reproducible, hermetic image. On merge to `main`, the Actions workflow pushes the image to GHCR (`ghcr.io/org/scene-gen:sha-<commit>`). Deployment pulls a specific immutable digest rather than `latest`, making rollbacks a one-line change. For robotics, deterministic environments matter because numerical libraries (numpy, torch) must match exactly between dev and robot hardware.

### 3. Pre-commit hooks with `ruff` + `mypy`
Running `ruff check` (fast linting) and `mypy --strict` as a pre-commit hook ensures that type errors and style violations are caught locally before CI ever runs. For a research codebase where multiple engineers modify shared utilities like `scene_generator.py`, this prevents silent breakages where a function's return type changes and downstream callers silently misuse it.

---

## FastAPI endpoint behaviour

- `POST /generate` validates the request via Pydantic, calls `scene_generator.generate()`, and returns a `GenerateResponse` with `configs`, `n_valid`, and `coverage_score`.
- Invalid strategy names or out-of-range `n` return HTTP 400 with a descriptive message.
- `GET /health` returns `{"status": "ok"}` for load-balancer health checks.
