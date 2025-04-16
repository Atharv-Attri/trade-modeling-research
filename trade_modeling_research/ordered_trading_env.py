from gym_trading_env.environments import MultiDatasetTradingEnv
import random
import pandas as pd

class OrderedMultiDatasetTradingEnv(MultiDatasetTradingEnv):
    def __init__(self, *args, shuffle_datasets=True, random_start=True, **kwargs):
        self.shuffle_datasets = shuffle_datasets
        self.random_start = random_start
        self.dataset_index = -1
        super().__init__(*args, **kwargs)

    def _choose_dataset(self):
        if self.shuffle_datasets:
            self.current_dataset_path = random.choice(self.dataset_pathes)
        else:
            self.dataset_index = (self.dataset_index + 1) % len(self.dataset_pathes)
            self.current_dataset_path = self.dataset_pathes[self.dataset_index]

        #print(f"[env] Loaded dataset: {self.current_dataset_path}")
        self._set_df(self.preprocess(pd.read_pickle(self.current_dataset_path)))

    def reset(self, **kwargs):
        self._choose_dataset()

        if self.random_start:
            max_start = len(self.df) - self.episode_length
            self.current_index = random.randint(0, max(0, max_start))
        else:
            self.current_index = 0

        return super().reset(**kwargs)
