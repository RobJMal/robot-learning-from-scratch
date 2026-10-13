import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results/pusht_dataset_stats")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(glob.glob("data/pusht_keypoints/data/chunk-*/file-*.parquet"))
df = pd.read_parquet(files[0])


def get_episode(df, idx):
    ep = df[df["episode_index"] == idx]
    state = np.stack(ep["observation.state"].to_numpy())                  # (T, 2)
    env_state = np.stack(ep["observation.environment_state"].to_numpy())  # (T, 16)
    actions = np.stack(ep["action"].to_numpy())                           # (T, 2)
    reward = ep["next.reward"].to_numpy()                                 # (T,)
    return state, env_state, actions, reward


state, env_state, actions, reward = get_episode(df, 0)
t = np.arange(len(state))

# Print ranges, so you can see what normalization will need to handle
for name, arr in [("state", state), ("env_state", env_state), ("action", actions)]:
    print(name, arr.shape, arr.min(), arr.max())

# --- Figure 1: the three main panels ---
fig, ax = plt.subplots(1, 3, figsize=(16, 4))

ax[0].plot(t, actions[:, 0], label="action x")
ax[0].plot(t, actions[:, 1], label="action y")
ax[0].set_title("Actions vs time")
ax[0].set_xlabel("timestep"); ax[0].legend()

ax[1].plot(t, actions[:, 0], "--", label="action x")
ax[1].plot(t, state[:, 0], label="state x")
ax[1].set_title("Action vs state (x)")
ax[1].set_xlabel("timestep"); ax[1].legend()

sc = ax[2].scatter(state[:, 0], state[:, 1], c=t, s=10, cmap="viridis")
ax[2].set_title("Agent path (color = time)")
ax[2].set_xlabel("x"); ax[2].set_ylabel("y")
ax[2].set_aspect("equal")
fig.colorbar(sc, ax=ax[2], label="timestep")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "episode0_overview.png", dpi=120)

# --- Figure 2: block keypoints and reward ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t, env_state)
ax[0].set_title("Block keypoints over time (16 values)")
ax[0].set_xlabel("timestep")
ax[1].plot(t, reward)
ax[1].set_title("next.reward over time")
ax[1].set_xlabel("timestep")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "episode0_keypoints_reward.png", dpi=120)

# --- Figure 3: paths from several episodes ---
plt.figure(figsize=(5, 5))
for idx in range(5):
    s, _, _, _ = get_episode(df, idx)
    plt.plot(s[:, 0], s[:, 1], label=f"episode {idx}")
plt.legend(); plt.gca().set_aspect("equal")
plt.title("Agent paths, 5 episodes")
plt.savefig(RESULTS_DIR / "paths_multi_episode.png", dpi=120)

plt.show()