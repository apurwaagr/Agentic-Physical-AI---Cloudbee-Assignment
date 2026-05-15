# Research Engineer Assignment — CloudBee Robotics

## Background

This repository contains my solution to the CloudBee Robotics Research Engineer take-home assignment. The work covers diffusion model debugging, low-discrepancy scene sampling, foundation model evaluation, and a robotic task agent with a FastAPI deployment.

## Setup Instructions

```bash
# Python 3.10 or 3.11 required
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4 matplotlib==3.9.0 pytest==8.2.0 \
            fastapi==0.115.0 uvicorn==0.30.0 pydantic==2.7.0
```

No GPU required. All tasks run on CPU.

### Part 3A: Configure LLM API (Optional)

To use the real LLM agent (Google Gemini), create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# GEMINI_API_KEY=your_api_key_here
# GEMINI_MODEL=gemini-2.5-flash
```

Get a free Gemini API key at [aistudio.google.com/app/apikeys](https://aistudio.google.com/app/apikeys).

**Note:** The `.env` file is in `.gitignore` — it won't be committed to GitHub. When you clone this repo, create your own `.env` locally.

---

## Running Each Task

### Part 1A — DDPM Scheduler (fixed)

```bash
cd part1
python test_1a.py        # all 4 tests pass
python ddpm_scheduler.py # prints snr values + sanity check passed
```

Three bugs were fixed — see `FIX [1/2/3]` comments in `ddpm_scheduler.py`.

### Part 1C — World Model

```bash
cd part1
python world_model.py    # prints: OK  shape: torch.Size([2, 10, 32])
```

### Part 2A — Scene Generator

```bash
cd part2
pytest test_2a.py        # 3/3 tests pass
python scene_generator.py  # saves plots/coverage.png
```

### Part 2B — Foundation Model Evaluation

```bash
cd part2
python eval_2b.py        # uses LightweightPolicy (offline, no downloads)
                         # saves plots/action_error.png
```

### Part 3A — Task Agent

```bash
cd part3
python agent.py          # writes execution_log.json
```

If a local Ollama instance is running at `http://localhost:11434`, the agent uses it for LLM calls. Otherwise it falls back to a deterministic built-in planner — the `execution_log.json` is always produced.

### Part 3B — FastAPI + Docker

```bash
cd part3
uvicorn app:app --reload --port 8000

# Test the endpoint:
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d "{\"task\":\"pick-and-place\",\"n\":10,\"strategy\":\"halton\"}"

# Docker (run from starter_kit/ root so COPY paths resolve):
cd ..
docker build -f part3/Dockerfile -t scene-gen .
docker run -p 8000:8000 scene-gen
```

---

## Task Summary

| Task | Status | Key files |
|------|--------|-----------|
| 1A DDPM bugs | ✅ All 3 fixed, 4/4 tests pass | `part1/ddpm_scheduler.py` |
| 1B Video diffusion | ✅ Written | `part1/answers.md` |
| 1C WorldModel | ✅ All 4 methods | `part1/world_model.py` |
| 2A Scene generator | ✅ Both stubs fixed, 3/3 tests | `part2/scene_generator.py` |
| 2B Foundation model | ✅ LightweightPolicy, plot saved | `part2/eval_2b.py` |
| 2C Sim-to-real | ✅ Written | `part2/sim_to_real.md` |
| 3A Agent | ✅ Full loop, replan on obj_003 failure | `part3/agent.py` |
| 3B FastAPI + Docker | ✅ Endpoint + Dockerfile complete | `part3/app.py`, `part3/Dockerfile` |

---

## Original Assignment Kit Contents

**CloudBee Robotics R&D  ·  Confidential  ·  Authorised candidates only**

---

## What Is In This Package

This ZIP is everything you need to start the assignment. Nothing is missing.

| File | Task | Your role |
|------|------|-----------|
| `part1/ddpm_scheduler.py` | 1A | Find & fix 3 bugs |
| `part1/test_1a.py` | 1A | Must all pass after your fixes |
| `part1/world_model.py` | 1C | Implement 4 stubbed methods |
| `part2/scene_generator.py` | 2A | Fix 2 broken stubs |
| `part2/test_2a.py` | 2A | Must all pass after your fixes |
| `part2/inference_example.py` | 2B | **Working example** — run this first (uses SmolVLA/ACT) |
| `part2/eval_2b.py` | 2B | Scaffold — fill in the TODOs |
| `part2/checkpoints/lightweight_policy.pt` | 2B | Fallback only — use if lerobot won't install |
| `part2/data/sample_episode.npz` | 2B | Fallback only — synthetic episode for LightweightPolicy |
| `part3/agent.py` | 3A | Scaffold — implement the agent loop |
| `part3/app.py` | 3B | Scaffold — implement the endpoint |
| `part3/Dockerfile` | 3B | Template — complete the Dockerfile |
| `requirements.txt` | All | Pinned dependencies |

Files you create from scratch: `part1/answers.md`, `part2/foundation_model_report.md`,
`part2/sim_to_real.md`, `part3/execution_log.json`, `part3/devops.md`, `part3/reflection.md`,
`bonus/` (optional).

---

## Setup

**Python version:** 3.10 or 3.11 (tested). Python 3.12 may have minor compatibility issues with some robotics libraries.

**No GPU required.** All tasks run on CPU.

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install pinned dependencies (lerobot is included — expect a 2-4 min install)
pip install -r requirements.txt

# 3. Verify Task 1A setup
cd part1
python test_1a.py               # should FAIL (bugs not yet fixed — that's expected)

# 4. Verify Task 2B setup — downloads SmolVLA/ACT model + pusht dataset on first run
cd ../part2
python inference_example.py     # runs inference, saves plots/action_error.png
```

**First run of `inference_example.py` will download ~200–500 MB** (model + dataset via HuggingFace).
Subsequent runs use the local cache.  If your internet is restricted, see the Fallback section below.

---

## Task-by-Task Quick Start

### Part 1A — DDPM Scheduler (debug)
```bash
cd part1
python test_1a.py               # find which tests fail → locate the bugs
# edit ddpm_scheduler.py, add FIX [n] comments
python test_1a.py               # all 4 tests must pass
```

### Part 1C — World Model (implement)
```bash
cd part1
python world_model.py           # must print "OK  shape: torch.Size([2, 10, 32])"
```

### Part 2A — Scene Generator (fix stubs)
```bash
cd part2
pytest test_2a.py               # find which tests fail → fix the stubs
python scene_generator.py       # generates configs + saves plots/coverage.png
```

### Part 2B — Foundation Model Evaluation
```bash
cd part2
python inference_example.py     # loads SmolVLA/ACT, runs inference, saves plot
# Then complete eval_2b.py:
#   - set MODEL_CHOICE = "smolvla"  or  "act"
#   - fill in the TODO blocks
python eval_2b.py               # saves plots/action_error.png
```

**Choosing a model:**
- **SmolVLA** (`lerobot/smolvla_base`) — recommended: compact modern VLA, current research direction.
- **ACT** (`lerobot/act_pusht_image`) — well-documented, standard lerobot example, easy to get working.
- **LightweightPolicy** (provided in ZIP) — fallback only, if lerobot fails to install.
  Set `MODEL_CHOICE = "lightweight"` in `eval_2b.py`. Works fully offline, no downloads.

All three options receive **equal marks**. The quality of your written analysis matters more than which model you pick.

### Part 3A — Task Agent
```bash
cd part3
# Set your LLM API key:
export ANTHROPIC_API_KEY="..."    # or OPENAI_API_KEY / GEMINI_API_KEY
# Implement agent.py, then:
python agent.py                   # must produce execution_log.json
python -c "import json; json.load(open('execution_log.json'))"  # must not raise
```

### Part 3B — FastAPI + Docker
```bash
cd part3
uvicorn app:app --reload --port 8000   # test locally
# Then complete the Dockerfile:
docker build -t scene-gen .
docker run -p 8000:8000 scene-gen
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"task":"pick-and-place","n":10,"strategy":"halton"}'
```

---

## Known Limitations & Troubleshooting

| Issue | Fix |
|-------|-----|
| `torch` import error | Run `pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: scene_generator` in Part 3B | Adjust `sys.path` in `app.py` to point to your `part2/` directory |
| `lerobot` install fails (CUDA/gym conflict) | Set `MODEL_CHOICE = "lightweight"` in `eval_2b.py` — uses the provided `.pt` + `.npz`, no download needed |
| SmolVLA model class not found | Switch to ACT: change `MODEL_ID = "lerobot/act_pusht_image"` in `inference_example.py` |
| HuggingFace download slow / blocked | Run `HF_HUB_OFFLINE=1 python eval_2b.py` after a first successful download to use cache |
| Docker build fails on ARM (M1/M2 Mac) | Add `--platform linux/amd64` to `docker build` |
| `pytest` not found | `pip install pytest==8.2.0` |

---

## Submission Format

Name your ZIP: **`YourName_Assignment.zip`**

Required structure:
```
YourName_Assignment/
├── README.md               ← your background + setup instructions
├── requirements.txt        ← any additional packages you added
├── part1/
│   ├── ddpm_scheduler.py   ← bugs fixed, FIX comments added
│   ├── world_model.py      ← all 4 methods implemented
│   └── answers.md          ← Task 1B written answers
├── part2/
│   ├── scene_generator.py  ← stubs fixed
│   ├── eval_2b.py          ← completed
│   ├── plots/              ← coverage.png + action_error.png
│   ├── configs/            ← generated JSON files
│   ├── foundation_model_report.md
│   └── sim_to_real.md
├── part3/
│   ├── agent.py            ← implemented
│   ├── execution_log.json  ← output from agent.py
│   ├── app.py              ← implemented
│   ├── Dockerfile          ← completed
│   ├── devops.md
│   └── reflection.md
└── bonus/                  ← optional
    ├── bonus.md
    └── <code files>
```

**Video (5–8 min screen recording):**
1. Walk through your FIX comments in `ddpm_scheduler.py` (2 min)
2. Show `scene_generator.py` + coverage plots (1–2 min)
3. Show `agent.py` handling the `obj_003` failure (2 min)

Upload to Google Drive (shared link) or YouTube (unlisted). Email ZIP + video link within 5 days.

---

## Scoring at a Glance

| Task | Points | What earns full marks |
|------|--------|-----------------------|
| 1A DDPM bugs | 25 | All 3 bugs found, explained, tests pass |
| 1B Video diffusion | 15 | Specific mechanisms, not vague generalities |
| 1C WorldModel | 20 | All 4 methods correct, docstring explains latent vs pixel |
| 2A Scene generator | 20 | Both stubs fixed, 3 tests pass, new param justified |
| 2B Foundation model | 20 | Written answers specific + error plot + peak analysis |
| 2C Sim-to-real | 10 | Two mechanisms named precisely, concrete mitigations |
| 3A Agent | 25 | Valid JSON log, failure detected, replanned correctly |
| 3B FastAPI + Docker | 20 | Endpoint works, Dockerfile builds, 3 CI tools named |
| README + Video | 15 | Clear setup, walkthrough covers required parts |
| **Bonus** | +15 | One option, well-reasoned |
| **Total** | **170** | |

**Prioritise correctness and reasoning over completeness.**
A partially implemented task with clear written reasoning scores substantially better
than a complete implementation with no explanation. If you cannot finish a section,
describe your approach and what you would do given more time.
