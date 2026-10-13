# Standard
import glob
import json

# Third-party
import pandas as pd
import numpy as np

DATA_GLOB = "data/pusht_keypoints/data/chunk-*/file-*.parquet"
DIV_ZERO_GUARD = 1e-06

def load_dataframe(pattern: str = DATA_GLOB) -> pd.DataFrame:
    print(f"[load_dataframe] Reading parquet files matching {pattern}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files match {pattern}")
    df = pd.read_parquet(files[0])
    print(f"[load_dataframe] Loaded {files[0]}: {df.shape[0]} rows, "
          f"{df['episode_index'].nunique()} episodes")
    return df

def get_episode(df: pd.DataFrame, episode_num: int):
    """
    Extracts the arrays from the dataframe and puts them into 
    numpy matrices (easier to conceptualize and deal with). 
    """
    ep_mask = df["episode_index"] == episode_num
    ep_df = df[ep_mask]

    state = np.stack(ep_df["observation.state"].to_numpy())                 # (T, 2)
    env_state = np.stack(ep_df["observation.environment_state"].to_numpy()) # (T, 16)
    actions = np.stack(ep_df["action"].to_numpy())                          # (T, 2)
    rewards = np.stack(ep_df["next.reward"].to_numpy())                     # (T)

    return state, env_state, actions, rewards

def main(): 
    df = load_dataframe()

    state = np.stack(df["observation.state"].to_numpy())                 # (T, 2)
    env_state = np.stack(df["observation.environment_state"].to_numpy()) # (T, 16)
    actions = np.stack(df["action"].to_numpy())                          # (T, 2)
    rewards = np.stack(df["next.reward"].to_numpy())                     # (T)

    state_mean, state_std = np.mean(state, axis=0), np.std(state, axis=0)
    env_state_mean, env_state_std = np.mean(env_state, axis=0), np.std(env_state, axis=0)
    actions_mean, actions_std = np.mean(actions, axis=0), np.std(actions, axis=0)
    rewards_mean, rewards_std = np.mean(rewards, axis=0), np.std(rewards, axis=0)

    json_data = {
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

    output_json_filepath = "data/normalized_data.json"
    with open(output_json_filepath, "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    main()
