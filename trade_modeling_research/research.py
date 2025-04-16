import pandas as pd
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium import Wrapper
import gym_trading_env
import numpy as np
import data_helper
import trends
from rich import print
import wrappers
import glob
from pathlib import Path
import os

from stable_baselines3 import A2C, PPO

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.monitor import Monitor


from rewards import emotion_reward_function, reward_euphoric_reckless, reward_confident_aware
import pandas as pd
from rewards import emotion_counts
from ordered_trading_env import OrderedMultiDatasetTradingEnv

class Research:
    def __init__(self, max_episode_duration=30, vectors=10):
        self.env = None
        self.max_episode_duration = max_episode_duration
        self.vectors = vectors

    def preprocess(self, df: pd.DataFrame):
        df.set_index("datetime", inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)

        df["feature_close"] = df["close"].pct_change()
        df["feature_open"] = df["open"] / df["close"]
        df["feature_high"] = df["high"] / df["close"]
        df["feature_low"] = df["low"] / df["close"]
        df["feature_volume"] = df["volume"]

        df.dropna(inplace=True)
        return df
   
    def make_wrapped_env(self):
        raw_env = gym.make(
            "MultiDatasetTradingEnv",
            dataset_dir="../data/raw_data/*.pkl",
            preprocess=self.preprocess,
            positions=[-1, 0, 1],
            trading_fees=0.0,
            borrow_interest_rate=0.0,
            initial_position=1,
            max_episode_duration=30,
            verbose=1,
        )
        return wrappers.StableEnvWrapper(raw_env) 

    def run_env(self, episodes=100):
        self.env = gym.make(
            "MultiDatasetTradingEnv",
            dataset_dir="../data/raw_data/*.pkl",
            preprocess=self.preprocess,
            positions=[-1, 0, 1],  # SHORT, OUT, LONG
            trading_fees=0.00 / 100,
            borrow_interest_rate=0.000 / 100,
            initial_position=1,
        )

        for _ in range(episodes):
            done, truncated = False, False
            observation, info = self.env.reset()

            while not done and not truncated:
                action = self.env.action_space.sample()
                observation, reward, done, truncated, info = self.env.step(action)

        self.env.save_for_render(dir="../render_logs")
        
    def filter_short_datasets(self, min_rows=15):
        dataset_paths = glob.glob("../data/raw_data/*.pkl")

        filtered_dir = "../data/filtered_out"
        os.makedirs(filtered_dir, exist_ok=True)

        for path in dataset_paths:
            df = pd.read_pickle(path)
            df = self.preprocess(df)

            if len(df) < min_rows:
                print(f"[yellow]Moved {Path(path).name} → filtered_out/ ({len(df)} rows)")
                target_path = os.path.join(filtered_dir, Path(path).name)
                os.rename(path, target_path)

            
    def run_vector_env(self, episodes=100, min_df=45):
        self.filter_short_datasets(min_df)
        self.env = SyncVectorEnv([self.make_wrapped_env for _ in range(3)])

        for episode in range(episodes):
            terminated = [False] * self.env.num_envs
            obs, info = self.env.reset()
            step_count = 0
            MAX_STEPS = 1000

            while not all(terminated) and step_count < MAX_STEPS:
                actions = [
                    self.env.single_action_space.sample() if not terminated[i] else 0
                    for i in range(self.env.num_envs)
                ]
                obs, reward, done, truncated, info = self.env.step(actions)

                # Update terminated status
                terminated = [t or d or tr for t, d, tr in zip(terminated, done, truncated)]

                print(f"step {step_count} | done: {done}, truncated: {truncated} | terminated: {terminated}")
                step_count += 1
                
    def train_with_a2c(self, total_timesteps=100_000):
        print("[bold green]Filtering datasets before training...")
        self.filter_short_datasets(min_rows=45)

        print("[bold green]Preparing environment for A2C training...")
        def make_env():
            raw_env = gym.make(
                "MultiDatasetTradingEnv",
                dataset_dir="../data/raw_data/*.pkl",
                preprocess=self.preprocess,
                positions=[-1, 0, 1],
                trading_fees=0.0,
                borrow_interest_rate=0.0,
                initial_position=1,
                max_episode_duration=30,
                verbose=0,
            )
            wrapped = wrappers.StableEnvWrapper(raw_env)
            return Monitor(wrapped)

        self.env = SubprocVecEnv([make_env for _ in range(5)])
        print(f"[bold blue]Training A2C for {total_timesteps:,} timesteps...")
        model = A2C("MlpPolicy", self.env, verbose=1)
        model.learn(total_timesteps=total_timesteps)

        os.makedirs("../models", exist_ok=True)
        model.save("../models/a2c_trading_model")
        print("[bold green]Model saved to [bold]/models/a2c_trading_model")
        print("[bold green]Model saved to [bold]/models/a2c_trading_model")

        self.evaluate_model(model, make_env)

    def train_with_ppo(self, total_timesteps=500_000, reward=emotion_reward_function):
        print("[bold green]Filtering datasets before training...")
        self.filter_short_datasets(min_rows=int(self.max_episode_duration *2))

        print("[bold green]Preparing environment for PPO training...")
        def make_env():
            raw_env = OrderedMultiDatasetTradingEnv(
                dataset_dir="../data/raw_data/*.pkl",
                preprocess=self.preprocess,
                positions=[-1.0, -0.5, 0.0, 0.5, 1.0],
                trading_fees=0.0,
                borrow_interest_rate=0.0,
                initial_position=1,
                max_episode_duration=96,
                shuffle_datasets=False,      
                random_start=False,          
                reward_function=reward,
                verbose=0,
            )
            wrapped = wrappers.StableEnvWrapper(raw_env)
            return Monitor(wrapped)

        self.env = SubprocVecEnv([make_env for _ in range(self.vectors)])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10.)

        print(f"[bold blue]Training PPO for {total_timesteps:,} timesteps...")

        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=0.0005,
            clip_range=0.15,
            ent_coef=0.02,
            n_steps=4096,        
            batch_size=512,       
            normalize_advantage=True,
        )

        model.learn(total_timesteps=total_timesteps)

        os.makedirs("../models", exist_ok=True)
        model.save(f"../models/ppo_{reward.__name__}")
        print(f"[bold green]Model saved to [bold]../models/ppo_{reward.__name__}")

        self.evaluate_model(model, make_env)
        print("\n[bold]Emotion Trigger Breakdown (cumulative):")
        total = sum(emotion_counts.values())
        for emotion, count in emotion_counts.items():
            pct = 100 * count / total if total > 0 else 0
            print(f"  • {emotion.title():<10} : {count} ({pct:.1f}%)")
            
        
    
    def evaluate_model(self, model, env_fn, n_eval_episodes=10):
        returns = []

        for _ in range(n_eval_episodes):
            env = env_fn()
            obs, info = env.reset()
            done, truncated = False, False
            initial_value = info.get("portfolio_valuation", 1.0)
            final_value = initial_value

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                final_value = info.get("portfolio_valuation", final_value)

            returns.append((final_value / initial_value) - 1)

        avg_return = np.mean(returns)
        print(f"\n[bold green]Final Evaluation: Avg Portfolio Return = {avg_return * 100:.2f}% over {n_eval_episodes} episodes")
    
    def get_data(self,days=5, tf="4h", spike=True):
        trends.get_top_10(days)
        trends.get_dates(spike)
        data_helper.get_dates(tf)

    def clear_render_logs(self):
        data_helper.clear_render_logs()
        
    def clear_cache(self):
        data_helper.clear_cache()
    
    def clear_raw_data(self):
        data_helper.clear_raw_data()
    
    def clear_filtered_out(self):
        data_helper.clear_filtered_out()
        
    def clear_all(self):
        self.clear_cache()
        self.clear_raw_data()
        self.clear_render_logs()
        self.clear_filtered_out()
        


if __name__ == "__main__":
    research = Research(max_episode_duration=96, vectors=12)
    #research.clear_all()
    #research.get_data(days=200, tf='5m', spike=True)
    research.train_with_ppo(total_timesteps=1_000_000, reward=reward_confident_aware)
