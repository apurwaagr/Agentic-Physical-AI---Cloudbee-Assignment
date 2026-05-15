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
#       Model:    lerobot/act_pusht_image
#       Dataset:  lerobot/pusht
#       Install:  pip install lerobot
#
#   Option C — LightweightPolicy  (fallback: no download, works offline)
#       Files provided: checkpoints/lightweight_policy.pt + data/sample_episode.npz
#       No extra install needed.
#       Use this ONLY if lerobot fails to install.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

# ── CONFIG — set before running ────────────────────────────────────────────────
MODEL_CHOICE = "lightweight"   # "smolvla" | "act" | "lightweight"
T_EVAL       = 50              # timesteps to evaluate (first T_EVAL of episode 0)

# ── 1. Load model and dataset ─────────────────────────────────────────────────

if MODEL_CHOICE == "smolvla":
    # TODO: load SmolVLA from HuggingFace
    # from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    # policy  = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    # policy.eval()
    # from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    # dataset = LeRobotDataset("lerobot/pusht")
    policy  = None   # TODO
    dataset = None   # TODO

elif MODEL_CHOICE == "act":
    # TODO: load ACT from HuggingFace
    # from lerobot.common.policies.act.modeling_act import ACTPolicy
    # policy  = ACTPolicy.from_pretrained("lerobot/act_pusht_image")
    # policy.eval()
    # from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    # dataset = LeRobotDataset("lerobot/pusht")
    policy  = None   # TODO
    dataset = None   # TODO

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
# TODO: load the first T_EVAL timesteps from episode 0.
#
# For SmolVLA / ACT (lerobot dataset):
#   episode_start = dataset.episode_data_index["from"][0].item()
#   frames = [dataset[episode_start + t] for t in range(T_EVAL)]
#   gt_actions = np.stack([f["action"].numpy() for f in frames])
#
# For LightweightPolicy:
#   episode  = np.load("data/sample_episode.npz", allow_pickle=True)
#   observations = episode["observations"][:T_EVAL]   # (T_EVAL, 8)
#   gt_actions   = episode["actions"][:T_EVAL]        # (T_EVAL, 2)
#
observations = None   # not used for lerobot; populated below for lightweight
gt_actions   = None   # TODO: numpy array (T_EVAL, act_dim)

if MODEL_CHOICE == "lightweight":
    episode      = np.load("data/sample_episode.npz", allow_pickle=True)
    observations = episode["observations"][:T_EVAL]   # (T_EVAL, 8)
    gt_actions   = episode["actions"][:T_EVAL]        # (T_EVAL, 2)
else:
    # SmolVLA / ACT lerobot path
    episode_start = dataset.episode_data_index["from"][0].item()
    frames        = [dataset[episode_start + t] for t in range(T_EVAL)]
    gt_actions    = np.stack([f["action"].numpy() for f in frames])

# ── 3. Run inference ──────────────────────────────────────────────────────────
if MODEL_CHOICE == "lightweight":
    pred_actions = np.stack([policy.predict(observations[t]) for t in range(T_EVAL)])
else:
    # SmolVLA / ACT
    if hasattr(policy, "reset"):
        policy.reset()
    pred_actions = []
    for t in range(T_EVAL):
        with torch.no_grad():
            pred = policy.select_action(frames[t])
        pred_actions.append(pred.numpy())
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

# ── 6. Written analysis (goes in foundation_model_report.md) ──────────────────
# TODO: Write 2-3 sentences in foundation_model_report.md answering:
#
#   "WHY is the error highest at t={peak_t}?"
#
#   Do NOT say "the model was uncertain."
#   Explain the specific mechanism — for example:
#   - Action chunking: ACT predicts K-step chunks; error compounds within a chunk
#   - Trajectory transitions: high curvature = sudden velocity change the model didn't see
#   - Distribution shift: if using LightweightPolicy, frames beyond t=47 were never in training
#   - Contact / object interaction: error spikes when the robot makes or breaks contact
