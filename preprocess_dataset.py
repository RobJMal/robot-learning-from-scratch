# Standard
import glob
import json

# Third-party
import pandas as pd
import numpy as np

DATA_GLOB = "data/pusht_keypoints/data/chunk-*/file-*.parquet"
RNG_SEED = 137
TRAIN_SPLIT = 0.8   # Between 0.0 to 1.0

def load_dataframe(pattern: str = DATA_GLOB) -> pd.DataFrame:
    print(f"[load_dataframe] Reading parquet files matching {pattern}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files match {pattern}")
    df = pd.read_parquet(files[0])
    print(f"[load_dataframe] Loaded {files[0]}: {df.shape[0]} rows, "
          f"{df['episode_index'].nunique()} episodes")
    return df

def main(): 
    df = load_dataframe()

    # Split training and validation
    episode_idxs = sorted(df["episode_index"].unique().tolist())
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(episode_idxs)

    train_val_split_idx = round(len(episode_idxs) * TRAIN_SPLIT)
    train_data_episode_idxs = episode_idxs[:train_val_split_idx]
    val_data_episode_idxs = episode_idxs[train_val_split_idx:]

    # Compute the mean and std for training data (important for normalization)
    train_eps_mask = df["episode_index"].isin(train_data_episode_idxs)
    train_eps_df = df[train_eps_mask]

    state = np.stack(train_eps_df["observation.state"].to_numpy())                 # (T, 2)
    env_state = np.stack(train_eps_df["observation.environment_state"].to_numpy()) # (T, 16)
    actions = np.stack(train_eps_df["action"].to_numpy())                          # (T, 2)
    rewards = np.stack(train_eps_df["next.reward"].to_numpy())                     # (T)

    state_mean, state_std = np.mean(state, axis=0), np.std(state, axis=0)
    env_state_mean, env_state_std = np.mean(env_state, axis=0), np.std(env_state, axis=0)
    actions_mean, actions_std = np.mean(actions, axis=0), np.std(actions, axis=0)
    rewards_mean, rewards_std = np.mean(rewards, axis=0), np.std(rewards, axis=0)

    json_data = {
        "dataset_split" : {
            "train": train_data_episode_idxs,
            "val": val_data_episode_idxs,
        },
        "state" : {
            "mean": state_mean.tolist(),
            "std": state_std.tolist(),
        },
        "env_state" : {
                    "mean": env_state_mean.tolist(),
                    "std": env_state_std.tolist(),
                },
        "actions" : {
                    "mean": actions_mean.tolist(),
                    "std": actions_std.tolist(),
                },
        "rewards" : {
                    "mean": rewards_mean.tolist(),
                    "std": rewards_std.tolist(),
                },
    }

    output_json_filepath = "config/normalized_data.json"
    with open(output_json_filepath, "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    main()
