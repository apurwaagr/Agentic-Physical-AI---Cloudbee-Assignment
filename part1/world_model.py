# world_model.py  -  Task 1C
# Implement all 4 stubbed methods using PyTorch.
# Do NOT change __init__ or the provided layer names.
# Replace the forward() docstring with exactly 2 sentences explaining WHY
# latent-space prediction is preferred over pixel-space prediction for robot tasks.

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """
    RSSM-inspired world model for sequential robot observations.
    obs_dim: raw observation size  |  act_dim: action size
    lat_dim: latent size           |  hid_dim: hidden width
    """
    def __init__(self, obs_dim, act_dim, lat_dim=64, hid_dim=128):
        super().__init__()
        self.encoder    = nn.Linear(obs_dim, lat_dim)
        self.transition = nn.GRUCell(act_dim, lat_dim)
        self.decoder    = nn.Linear(lat_dim, obs_dim)

    def encode(self, obs):
        """Encode a raw observation into a latent vector."""
        return self.encoder(obs)

    def transition_step(self, z, a):
        """Predict next latent state given current latent z and action a."""
        return self.transition(a, z)

    def decode(self, z):
        """Decode a latent vector back to observation space."""
        return self.decoder(z)

    def forward(self, obs_seq, act_seq):
        """
        Latent-space prediction is preferred because the latent space is compact and
        structured, forcing the model to learn task-relevant features rather than
        wasting capacity reconstructing task-irrelevant pixel details. Predicting in
        a low-dimensional latent space also makes dynamics learning more tractable and
        numerically stable, which directly improves downstream planning and control.

        obs_seq: (B, T, obs_dim)   act_seq: (B, T, act_dim)
        Returns: pred_obs  (B, T, obs_dim)
        """
        B, T, _ = obs_seq.shape
        # Encode all observations at once: (B, T, lat_dim)
        z = self.encode(obs_seq.view(B * T, -1)).view(B, T, -1)

        pred_obs = []
        # Use z[:, 0, :] as initial latent state; roll through T steps
        z_t = z[:, 0, :]                        # (B, lat_dim)
        for t in range(T):
            pred_obs.append(self.decode(z_t))   # decode current latent
            z_t = self.transition_step(z_t, act_seq[:, t, :])  # step forward

        return torch.stack(pred_obs, dim=1)     # (B, T, obs_dim)


if __name__ == "__main__":
    m = WorldModel(obs_dim=32, act_dim=4)
    obs = torch.randn(2, 10, 32)
    act = torch.randn(2, 10,  4)
    out = m(obs, act)
    assert out.shape == (2, 10, 32), f"Got {out.shape}"
    print("OK  shape:", out.shape)
