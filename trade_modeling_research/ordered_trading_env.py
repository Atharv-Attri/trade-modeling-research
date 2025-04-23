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
        # pick your start however you like…
        if self.random_start:
            max_start = len(self.df) - self.max_episode_duration
            self.current_index = random.randint(0, max(0, max_start))
        else:
            self.current_index = 0

        # now call the parent — it WILL randomize _idx 
        obs, info = super().reset(**kwargs)

        # but immediately force it back to chosen start
        self._idx = self.current_index

        return obs, info
    
    
    def _set_df(self, df):
        df = df.copy()
        
        if "close" not in df.columns and "data_close" in df.columns:
            df["close"] = df["data_close"]

        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(set(list(df.columns)) - set(self._features_columns))

        if "close" not in self._info_columns:
            self._info_columns.append("close")

        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        for i, dynamic_func in enumerate(self.dynamic_feature_functions):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = df[self._features_columns].values.astype("float32")
        self._info_array = df[self._info_columns].values
        self._price_array = df["close"].values

