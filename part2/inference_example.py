# inference_example.py  —  Task 2B: Working Offline Evaluation Example
# ─────────────────────────────────────────────────────────────────────────────
# Run this FIRST to verify your environment before writing eval_2b.py.
#
#   pip install lerobot
#   python inference_example.py
#
# This script loads lerobot/act_pusht_image and runs offline inference on
# the first 50 timesteps of the lerobot/pusht dataset.
# No robot, no simulator, no environment required — just forward passes on
# recorded observations.
#
# If lerobot fails to install (CUDA mismatch, network issues, etc.) see the
# FALLBACK section at the bottom of this file.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Change MODEL_ID to try SmolVLA:
#   MODEL_ID  = "lerobot/smolvla_base"
#   POLICY_CLS = "SmolVLAPolicy"
MODEL_ID   = "lerobot/act_pusht_image"
DATASET_ID = "lerobot/pusht"
T_EVAL     = 50    # number of timesteps to evaluate

# ── 1. Load model ─────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_ID}")
try:
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(MODEL_ID)
    policy.eval()
    print(f"  Model loaded.  Parameters: {sum(p.numel() for p in policy.parameters()):,}")
except ImportError:
    raise SystemExit(
        "\nlerobot is not installed.\n"
        "Run:  pip install lerobot\n"
        "Or see the FALLBACK section at the bottom of this file."
    )

# ── 2. Load dataset ───────────────────────────────────────────────────────────
print(f"Loading dataset: {DATASET_ID}")
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset   = LeRobotDataset(DATASET_ID)
episode_0 = dataset.episode_data_index["from"][0].item()
episode_1 = dataset.episode_data_index["to"][0].item()
print(f"  Dataset loaded.  Episode 0: frames {episode_0}–{episode_1}")

# ── 3. Run inference on each timestep ────────────────────────────────────────
print(f"Running inference on first {T_EVAL} timesteps ...")

pred_actions = []
gt_actions   = []

# Reset the policy's recurrent state before starting the episode
if hasattr(policy, "reset"):
    policy.reset()

for t in range(min(T_EVAL, episode_1 - episode_0)):
    frame = dataset[episode_0 + t]

    # Ground-truth action at this timestep
    gt_actions.append(frame["action"].numpy())

    # Model inference  —  select_action returns a (act_dim,) tensor
    with torch.no_grad():
        pred = policy.select_action(frame)
    pred_actions.append(pred.numpy())

pred_actions = np.array(pred_actions)   # (T, act_dim)
gt_actions   = np.array(gt_actions)     # (T, act_dim)
T_actual     = len(pred_actions)

# ── 4. Compute L1 error ────────────────────────────────────────────────────────
l1_errors = np.abs(pred_actions - gt_actions).mean(axis=1)   # (T,)

mean_err = float(l1_errors.mean())
std_err  = float(l1_errors.std())
peak_t   = int(np.argmax(l1_errors))

print()
print(f"Mean L1 error:  {mean_err:.4f}")
print(f"Std  L1 error:  {std_err:.4f}")
print(f"Peak L1 error:  {l1_errors[peak_t]:.4f}  at t={peak_t}")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
pathlib.Path("plots").mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(l1_errors, color="#0F3460", linewidth=1.5, label="L1 error per timestep")
ax.axvline(peak_t, color="orange", linestyle=":", linewidth=1.2,
           label=f"Peak error  t={peak_t}  ({l1_errors[peak_t]:.3f})")
ax.set_xlabel("Timestep")
ax.set_ylabel("Mean L1 Error")
ax.set_title(f"Offline Action Prediction — {MODEL_ID.split('/')[-1]}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/action_error.png", dpi=150)
print("Saved: plots/action_error.png")

print()
print("─" * 60)
print("WHY is error highest at that region?")
print("  Write your 2-3 sentence answer in foundation_model_report.md")
print("  Hint: think about action chunking, compounding errors, and")
print("  what happens at transition points in the trajectory.")
print("─" * 60)


# ── FALLBACK: LightweightPolicy (no download required) ────────────────────────
# If lerobot installation fails, you can use the provided synthetic policy:
#
# import torch.nn as nn
# class LightweightPolicy(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(8, 32), nn.Tanh(),
#             nn.Linear(32, 32), nn.Tanh(),
#             nn.Linear(32, 2),
#         )
#     def forward(self, obs): return self.net(obs)
#     def predict(self, obs_np):
#         with torch.no_grad():
#             return self(torch.FloatTensor(obs_np)).numpy()
#
# ckpt = torch.load("checkpoints/lightweight_policy.pt", map_location="cpu")
# policy = LightweightPolicy()
# policy.load_state_dict(ckpt["model_state_dict"])
# policy.eval()
#
# episode = np.load("data/sample_episode.npz", allow_pickle=True)
# observations = episode["observations"]   # (60, 8)
# gt_actions   = episode["actions"]        # (60, 2)
# pred_actions = np.stack([policy.predict(observations[t]) for t in range(len(observations))])
# ─────────────────────────────────────────────────────────────────────────────
