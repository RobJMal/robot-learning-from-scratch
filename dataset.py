
import json
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from utils import load_dataframe, normalize


class PushTDataset(Dataset):
    def __init__(self, is_training: bool) -> None:
        super().__init__()
        pusht_dataset_filepath = "data/pusht_keypoints/data/chunk-*/file-*.parquet"
        df = load_dataframe(pusht_dataset_filepath)

        with open("config/normalized_data.json") as f:
            self.dataset_info = json.load(f)

        self.eps_idxs = None
        if is_training:
            self.eps_idxs = self.dataset_info["dataset_split"]["train"]
        else: 
            self.eps_idxs = self.dataset_info["dataset_split"]["val"]

        # Extracting the training data stats
        self.state_mean, self.state_std = np.array(self.dataset_info['state']['mean']), np.array(self.dataset_info['state']['std'])
        self.env_state_mean, self.env_state_std = np.array(self.dataset_info['env_state']['mean']), np.array(self.dataset_info['env_state']['std'])
        self.actions_mean, self.actions_std = np.array(self.dataset_info['actions']['mean']), np.array(self.dataset_info['actions']['std'])

        eps_mask = df["episode_index"].isin(self.eps_idxs)
        self.eps_df = df[eps_mask]
        self.ep_frame_idxs = self.eps_df[['episode_index', 'frame_index']].to_numpy().tolist()

    def __len__(self):
        return len(self.ep_frame_idxs)

    def __getitem__(self, index) -> Any:
        ep_idx, frame_idx = self.ep_frame_idxs[index]
        df = self.eps_df[(self.eps_df['episode_index']==ep_idx)&(self.eps_df['frame_index']==frame_idx)]

        # Creating the observation (18,)
        state = np.stack(df['observation.state'].to_numpy()) 
        state_normed = normalize(
            state, 
            mean=self.state_mean,
            std=self.state_std,
        )
        env_state = np.stack(df['observation.environment_state'].to_numpy())
        env_state_normed = normalize(
            env_state,
            mean=self.env_state_mean,
            std=self.env_state_std,
        )
        obs = np.concatenate([state_normed, env_state_normed], axis=1).flatten()
        breakpoint()

        # Creating the K actions to return (K, 2)


if __name__ == "__main__":
    training_dataset = PushTDataset(is_training=True)
    training_dataset[1]
