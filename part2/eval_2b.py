# eval_2b.py  —  Task 2B: Offline Action Prediction Evaluation
# ─────────────────────────────────────────────────────────────────────────────
# INSTRUCTIONS:
#   1. Run inference_example.py first — it shows a complete working example.
#   2. Choose ONE model below and implement the TODO blocks.
#   3. Run:  python eval_2b.py
#   4. Answer the written questions in foundation_model_report.md.
#
# MODEL OPTIONS (all receive equal marks):
#
#   Option A — SmolVLA  (recommended: small, modern VLA)
#       Model:    lerobot/smolvla_base
#       Dataset:  lerobot/pusht  (or lerobot/smolvla_base's own demo dataset)
#       Install:  pip install lerobot
#
#   Option B — ACT  (well-documented, standard lerobot example)
#       Model:    vvrs/act-pusht by default (override with ACT_MODEL_ID)
#       Dataset:  lerobot/pusht
#       Install:  pip install lerobot
#
#   Option C — LightweightPolicy  (fallback: no download, works offline)
#       Files provided: checkpoints/lightweight_policy.pt + data/sample_episode.npz
#       No extra install needed.
#       Use this ONLY if lerobot fails to install.
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import torch
import pathlib

os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(".matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG — set before running ────────────────────────────────────────────────
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "act")   # "smolvla" | "act" | "lightweight"
T_EVAL       = int(os.getenv("T_EVAL", "50"))      # first T_EVAL timesteps of episode 0
MODEL_IDS = {
    "smolvla": "lerobot/smolvla_base",
    # The original scaffold referenced `lerobot/act_pusht_image`, but that
    # repo is not public. `vvrs/act-pusht` is a public ACT policy trained on PushT.
    "act": os.getenv("ACT_MODEL_ID", "vvrs/act-pusht"),
}
DATASET_ID = os.getenv("DATASET_ID", "lerobot/pusht")
VIDEO_BACKEND = os.getenv("VIDEO_BACKEND", "pyav")


def _to_numpy(value):
    """Convert tensors or tensor-like values returned by LeRobot to numpy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _load_lerobot_policy(model_choice):
    """Load a real LeRobot policy and PushT dataset for offline inference."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "lerobot is not installed or its dataset API changed. Install it with `pip install lerobot`, "
            "or run `MODEL_CHOICE=lightweight python eval_2b.py` only as a fallback."
        ) from exc

    if model_choice == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy_cls = SmolVLAPolicy
    elif model_choice == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy_cls = ACTPolicy
    else:
        raise ValueError(f"Unsupported LeRobot policy: {model_choice!r}")

    model_id = MODEL_IDS[model_choice]
    print(f"Loading policy: {model_id}")
    policy = policy_cls.from_pretrained(model_id)
    policy.eval()

    print(f"Loading real dataset: {DATASET_ID}  (video_backend={VIDEO_BACKEND})")
    dataset = LeRobotDataset(DATASET_ID, video_backend=VIDEO_BACKEND)
    return policy, dataset


def _batch_lerobot_frame(frame, device):
    """Add a batch dimension to tensor observations before policy inference."""
    batch = {}
    for key, value in frame.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        else:
            batch[key] = value
    return batch

# ── 1. Load model and dataset ─────────────────────────────────────────────────

if MODEL_CHOICE == "smolvla":
    policy, dataset = _load_lerobot_policy("smolvla")

elif MODEL_CHOICE == "act":
    policy, dataset = _load_lerobot_policy("act")

elif MODEL_CHOICE == "lightweight":
    # Fallback: synthetic policy + provided episode — no download needed
    import torch.nn as nn

    class LightweightPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, 2),
            )
        def forward(self, obs):
            return self.net(obs)
        def predict(self, obs_np):
            with torch.no_grad():
                return self(torch.FloatTensor(obs_np)).numpy()
        def select_action(self, obs_np):
            return self.predict(obs_np)

    ckpt    = torch.load("checkpoints/lightweight_policy.pt", map_location="cpu")
    policy  = LightweightPolicy()
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    dataset = None   # not used for lightweight option

else:
    raise ValueError(f"Unknown MODEL_CHOICE: {MODEL_CHOICE!r}")

assert policy is not None, "Load a model before running"

# ── 2. Load episode data ──────────────────────────────────────────────────────
# For SmolVLA / ACT (lerobot dataset):
#   use the first T_EVAL rows from episode_index == 0.
#
# For LightweightPolicy:
#   episode  = np.load("data/sample_episode.npz", allow_pickle=True)
#   observations = episode["observations"][:T_EVAL]   # (T_EVAL, 8)
#   gt_actions   = episode["actions"][:T_EVAL]        # (T_EVAL, 2)
#
observations = None   # not used for lerobot; populated below for lightweight
gt_actions   = None   # numpy array (T_EVAL, act_dim)

if MODEL_CHOICE == "lightweight":
    episode      = np.load("data/sample_episode.npz", allow_pickle=True)
    observations = episode["observations"][:T_EVAL]   # (T_EVAL, 8)
    gt_actions   = episode["actions"][:T_EVAL]        # (T_EVAL, 2)
else:
    # SmolVLA / ACT lerobot path
    episode_indices = np.array([_to_numpy(x).item() for x in dataset.hf_dataset["episode_index"]])
    episode_start   = int(np.flatnonzero(episode_indices == 0)[0])
    episode_end     = int(np.flatnonzero(episode_indices == 0)[-1]) + 1
    T_EVAL          = min(T_EVAL, episode_end - episode_start)
    frames          = [dataset[episode_start + t] for t in range(T_EVAL)]
    gt_actions      = np.stack([_to_numpy(f["action"]) for f in frames])

# ── 3. Run inference ──────────────────────────────────────────────────────────
if MODEL_CHOICE == "lightweight":
    pred_actions = np.stack([policy.predict(observations[t]) for t in range(T_EVAL)])
else:
    # SmolVLA / ACT
    if hasattr(policy, "reset"):
        policy.reset()
    device = next(policy.parameters()).device
    pred_actions = []
    for t in range(T_EVAL):
        with torch.no_grad():
            pred = policy.select_action(_batch_lerobot_frame(frames[t], device))
        pred_actions.append(_to_numpy(pred).squeeze(0))
    pred_actions = np.array(pred_actions)

# ── 4. Compute L1 error per timestep ─────────────────────────────────────────
l1_errors = np.abs(pred_actions - gt_actions).mean(axis=1)   # (T_EVAL,)

mean_err = float(l1_errors.mean())
std_err  = float(l1_errors.std())
peak_t   = int(np.argmax(l1_errors))

print(f"Model:          {MODEL_CHOICE}")
print(f"Mean L1 error:  {mean_err:.4f}")
print(f"Std  L1 error:  {std_err:.4f}")
print(f"Peak L1 error:  {l1_errors[peak_t]:.4f}  at t={peak_t}")

# ── 5. Plot and save ──────────────────────────────────────────────────────────
pathlib.Path("plots").mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(l1_errors, color="#0F3460", linewidth=1.5, label="L1 error")
ax.axvline(peak_t, color="orange", linestyle=":", linewidth=1.2,
           label=f"Peak  t={peak_t}  ({l1_errors[peak_t]:.3f})")
ax.set_xlabel("Timestep")
ax.set_ylabel("Mean L1 Error")
ax.set_title(f"Offline Action Prediction Error — {MODEL_CHOICE}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/action_error.png", dpi=150)
print("Saved: plots/action_error.png")

# See foundation_model_report.md for the peak-error analysis.
