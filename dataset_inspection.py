import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed; we only save files
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_GLOB = "data/pusht_keypoints/data/chunk-*/file-*.parquet"
RESULTS_DIR = Path("results/pusht_dataset_stats")
FPS = 10

# Hypothesis: the 8 keypoints are two 4-point rectangles (the T's top bar, then its stem).
# Verify with plot_keypoint_labels(); if wrong, only this list needs to change.
BLOCK_POLYGONS = [[0, 1, 2, 3], [4, 5, 6, 7]]


# ---------- Loading ----------

def load_dataframe(pattern: str = DATA_GLOB) -> pd.DataFrame:
    print(f"[load_dataframe] Reading parquet files matching {pattern}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files match {pattern}")
    df = pd.read_parquet(files[0])
    print(f"[load_dataframe] Loaded {files[0]}: {df.shape[0]} rows, "
          f"{df['episode_index'].nunique()} episodes")
    return df


def get_episode(df: pd.DataFrame, idx: int):
    ep = df[df["episode_index"] == idx]
    state = np.stack(ep["observation.state"].to_numpy())                  # (T, 2)
    env_state = np.stack(ep["observation.environment_state"].to_numpy())  # (T, 16)
    actions = np.stack(ep["action"].to_numpy())                           # (T, 2)
    reward = ep["next.reward"].to_numpy()                                 # (T,)
    return state, env_state, actions, reward


# ---------- Text summaries ----------

def print_ranges(df: pd.DataFrame, idx: int = 0) -> None:
    """Print shape and min/max for one episode (min/max are over all columns)."""
    print(f"[print_ranges] Printing shape and min/max of state, env_state and action "
          f"for episode {idx}")
    state, env_state, actions, _ = get_episode(df, idx)
    for name, arr in [("state", state), ("env_state", env_state), ("action", actions)]:
        print(f"  {name}", arr.shape, arr.min(), arr.max())


# ---------- Static plots ----------

def plot_episode_overview(df: pd.DataFrame, idx: int, out_dir: Path) -> None:
    print(f"[plot_episode_overview] Plotting for episode {idx}: (1) actions vs time, "
          f"(2) action x vs state x, (3) 2D agent path colored by time")
    state, _, actions, _ = get_episode(df, idx)
    t = np.arange(len(state))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    ax[0].plot(t, actions[:, 0], label="action x")
    ax[0].plot(t, actions[:, 1], label="action y")
    ax[0].set_title("Actions vs time")
    ax[0].set_xlabel("timestep")
    ax[0].legend()

    ax[1].plot(t, actions[:, 0], "--", label="action x")
    ax[1].plot(t, state[:, 0], label="state x")
    ax[1].set_title("Action vs state (x)")
    ax[1].set_xlabel("timestep")
    ax[1].legend()

    sc = ax[2].scatter(state[:, 0], state[:, 1], c=t, s=10, cmap="viridis")
    ax[2].set_title("Agent path (color = time)")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[2].set_aspect("equal")
    fig.colorbar(sc, ax=ax[2], label="timestep")

    fig.tight_layout()
    path = out_dir / f"episode{idx}_overview.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def plot_keypoints_and_reward(df: pd.DataFrame, idx: int, out_dir: Path) -> None:
    print(f"[plot_keypoints_and_reward] Plotting for episode {idx}: (1) all 16 block "
          f"keypoint values over time, (2) next.reward over time")
    _, env_state, _, reward = get_episode(df, idx)
    t = np.arange(len(env_state))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, env_state)
    ax[0].set_title("Block keypoints over time (16 values)")
    ax[0].set_xlabel("timestep")
    ax[1].plot(t, reward)
    ax[1].set_title("next.reward over time")
    ax[1].set_xlabel("timestep")

    fig.tight_layout()
    path = out_dir / f"episode{idx}_keypoints_reward.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def plot_multi_episode_paths(df: pd.DataFrame, episode_ids, out_dir: Path) -> None:
    episode_ids = list(episode_ids)
    print(f"[plot_multi_episode_paths] Plotting 2D agent paths of episodes "
          f"{episode_ids} on one set of axes")
    fig, ax = plt.subplots(figsize=(5, 5))
    for idx in episode_ids:
        s, _, _, _ = get_episode(df, idx)
        ax.plot(s[:, 0], s[:, 1], label=f"episode {idx}")
    ax.legend()
    ax.set_aspect("equal")
    ax.set_title(f"Agent paths, {len(episode_ids)} episodes")

    path = out_dir / "paths_multi_episode.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def plot_keypoint_labels(df: pd.DataFrame, idx: int, frame: int, out_dir: Path) -> None:
    """Scatter the 8 keypoints of one frame, labeled by index, to work out their order."""
    print(f"[plot_keypoint_labels] Plotting the 8 block keypoints of episode {idx}, "
          f"frame {frame}, each labeled with its index (use this to check BLOCK_POLYGONS)")
    _, env_state, _, _ = get_episode(df, idx)
    kp = env_state[frame].reshape(8, 2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(kp[:, 0], kp[:, 1])
    for i, (x, y) in enumerate(kp):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=12)
    ax.set_aspect("equal")
    ax.set_title(f"keypoint indices, episode {idx}, frame {frame}")

    path = out_dir / f"keypoint_labels_ep{idx}_f{frame}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


# ---------- Video ----------

def make_episode_video(df: pd.DataFrame, idx: int, path: Path, fps: int = FPS) -> None:
    """Draw pusher, block keypoints and action target for one episode.
    Writes MP4 if ffmpeg is available (path ends in .mp4), otherwise a GIF."""
    print(f"[make_episode_video] Animating episode {idx}: block outline (keypoints), "
          f"pusher position, action target and pusher trail, at {fps} fps")
    state, env_state, actions, reward = get_episode(df, idx)
    T = len(state)
    kp = env_state.reshape(T, 8, 2)  # 8 keypoints, (x, y) each

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect("equal")

    block_lines = [
        ax.plot([], [], "-o", ms=4, color="tab:blue",
                label="block keypoints" if j == 0 else "_nolegend_")[0]
        for j in range(len(BLOCK_POLYGONS))
    ]
    pusher, = ax.plot([], [], "o", ms=12, color="tab:red", label="pusher (state)")
    target, = ax.plot([], [], "x", ms=8, color="black", label="action target")
    trail, = ax.plot([], [], "-", lw=1, color="tab:red", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7)

    def update(i):
        for line, poly in zip(block_lines, BLOCK_POLYGONS):
            pts = kp[i][poly + poly[:1]]  # close each polygon
            line.set_data(pts[:, 0], pts[:, 1])
        pusher.set_data([state[i, 0]], [state[i, 1]])
        target.set_data([actions[i, 0]], [actions[i, 1]])
        trail.set_data(state[: i + 1, 0], state[: i + 1, 1])
        ax.set_title(f"episode {idx}  step {i}/{T - 1}  reward {reward[i]:.2f}")
        return (*block_lines, pusher, target, trail)

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps)
    if path.suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps)
    else:
        writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=100)
    plt.close(fig)
    print(f"  saved {path}")


def save_episode_video(df: pd.DataFrame, idx: int, out_dir: Path) -> Path:
    """Pick MP4 or GIF depending on whether ffmpeg is installed."""
    print(f"[save_episode_video] Choosing output format for the episode {idx} animation")
    if animation.writers.is_available("ffmpeg"):
        path = out_dir / f"episode{idx}.mp4"
    else:
        print("  ffmpeg not found, writing a GIF instead")
        path = out_dir / f"episode{idx}.gif"
    make_episode_video(df, idx, path)
    return path


# ---------- Frequency analysis ----------

def compute_action_spectrum(df: pd.DataFrame, n_fft: int = 256, fps: int = FPS):
    """Average power spectrum of actions over all episodes.
    n_fft must be >= the longest episode (246 here)."""
    print(f"[compute_action_spectrum] Computing the FFT power spectrum of actions "
          f"(x and y) for every episode, then averaging (n_fft={n_fft}, fps={fps})")
    freqs = np.fft.rfftfreq(n_fft, d=1 / fps)  # 0 .. fps/2 Hz
    powers = []
    for idx in sorted(df["episode_index"].unique()):
        _, _, a, _ = get_episode(df, idx)
        a = a - a.mean(axis=0)                     # remove DC offset
        a = a * np.hanning(len(a))[:, None]        # window against leakage
        spec = np.abs(np.fft.rfft(a, n=n_fft, axis=0)) ** 2 / len(a)
        powers.append(spec)
    return freqs, np.mean(powers, axis=0)          # (n_fft//2+1, 2)


def plot_action_fft(df: pd.DataFrame, out_dir: Path, cutoffs=(0.5, 1.0, 2.0)) -> None:
    print("[plot_action_fft] Plotting (1) average action power spectrum on a log scale, "
          "(2) cumulative fraction of power below each frequency")
    freqs, mean_power = compute_action_spectrum(df)
    total = mean_power[1:].sum(axis=0)
    cum = np.cumsum(mean_power[1:], axis=0) / total

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d, label in enumerate(["action x", "action y"]):
        ax[0].semilogy(freqs[1:], mean_power[1:, d], label=label)
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("mean power")
    ax[0].set_title("Action spectrum, averaged over episodes")
    ax[0].legend()

    for d, label in enumerate(["x", "y"]):
        ax[1].plot(freqs[1:], cum[:, d], label=label)
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("cumulative power fraction")
    ax[1].set_title("Power below each frequency")
    ax[1].legend()

    fig.tight_layout()
    path = out_dir / "action_fft.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")

    for cutoff in cutoffs:
        k = freqs[1:] <= cutoff
        frac = mean_power[1:][k].sum(axis=0) / total
        print(f"  power below {cutoff} Hz (x, y):", frac.round(3))


# ---------- Entry point ----------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataframe()

    print_ranges(df, idx=0)
    plot_episode_overview(df, idx=0, out_dir=RESULTS_DIR)
    plot_keypoints_and_reward(df, idx=0, out_dir=RESULTS_DIR)
    plot_multi_episode_paths(df, episode_ids=range(5), out_dir=RESULTS_DIR)
    plot_keypoint_labels(df, idx=0, frame=0, out_dir=RESULTS_DIR)
    save_episode_video(df, idx=0, out_dir=RESULTS_DIR)
    plot_action_fft(df, out_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()