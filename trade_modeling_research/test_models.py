import os
import glob
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ordered_trading_env import OrderedMultiDatasetTradingEnv
from wrappers import StableEnvWrapper
from rich import print
import numpy as np
from typing import List, Tuple


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index("datetime", inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)

    # Ensure 'close' exists
    if "close" not in df.columns and "data_close" in df.columns:
        df["close"] = df["data_close"]
    if "close" not in df.columns:
        raise ValueError("'close' column is required in data")

    # Feature engineering
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"]  = df["open"]  / df["close"]
    df["feature_high"]  = df["high"]  / df["close"]
    df["feature_low"]   = df["low"]   / df["close"]
    df.dropna(inplace=True)
    return df

def make_env_for_file(filepath: str):
    env = OrderedMultiDatasetTradingEnv(
        dataset_dir=filepath,
        preprocess=preprocess,
        positions=[-1.0, -0.5, 0.0, 0.5, 1.0],
        trading_fees=0.0,
        borrow_interest_rate=0.0,
        initial_position=1,
        max_episode_duration='max',
        shuffle_datasets=False,
        random_start=False,
        reward_function=lambda *args, **kwargs: 0.0,
        verbose=0,
    )
    return StableEnvWrapper(env)

def main():
    test_files = glob.glob("../data/test_data/*.pkl")
    if not test_files:
        print("[bold red]No test files found in ../data/test_data/[/bold red]")
        exit()

    # Filter out too-short files
    MIN_ROWS = 96
    valid_files = []
    for path in test_files:
        df = pd.read_pickle(path)
        df = preprocess(df)
        if len(df) >= MIN_ROWS:
            valid_files.append(path)
        else:
            print(f"[yellow]Skipping {Path(path).name}: only {len(df)} rows[/yellow]")
    if not valid_files:
        print("[bold red]No valid test files.[/bold red]")
        exit()

    # Models and positions
    models = {
        "euphoric_reckless":   "../models/ppo_reward_euphoric_reckless.zip",
        "confident_aware":     "../models/ppo_reward_confident_aware.zip",
        "risk_averse_mindful": "../models/ppo_reward_risk_averse_mindful.zip",
    }
    positions = [-1.0, -0.5, 0.0, 0.5, 1.0]

    os.makedirs("../render_logs/model_test_results", exist_ok=True)

    for name, model_path in models.items():
        print(f"\n[bold cyan]Evaluating model:[/bold cyan] {name}")
        model = PPO.load(model_path)

        for file_path in valid_files:
            fname = Path(file_path).stem
            print(f"[blue] → File:[/blue] {fname}")

            # Load and preprocess data
            df_raw = pd.read_pickle(file_path)
            df_proc = preprocess(df_raw).reset_index()

            # Build environment and reset to get initial info
            env = make_env_for_file(file_path)
            obs, info = env.reset()
            V0 = info.get("portfolio_valuation", 1.0)

            records = []
            # Record step 0 (initial state)
            row0 = df_proc.iloc[0]
            # Initial position is set by the environment (1.0)
            records.append({
                "step": 0,
                "open": row0["open"],
                "high": row0["high"],
                "low": row0["low"],
                "close": row0["close"],
                "action": 1.0,  # Initial position
                "cumulative_return": 0.0
            })

            done, truncated = False, False
            step = 1
            while not (done or truncated) and step < len(df_proc):
                # Add batch dimension for model prediction
                action, _ = model.predict(np.expand_dims(obs, axis=0), deterministic=True)
                action = action[0]  # Extract single action
                obs, reward, done, truncated, info = env.step(action)
                
                Vt = info.get("portfolio_valuation", V0)
                cum_frac = (Vt / V0) - 1.0

                row = df_proc.iloc[step]
                records.append({
                    "step": step,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "action": positions[action],
                    "cumulative_return": cum_frac,
                })
                step += 1

            # Print and save
            final_valuation = info.get("portfolio_valuation", V0)
            total_frac = (final_valuation / V0) - 1.0
            total_pct = total_frac * 100.0
            print(f"  [green]Total return:[/green] {total_frac:.6f} ({total_pct:.2f}%)")

            out = pd.DataFrame.from_records(records)
            out_path = f"../render_logs/model_test_results/{name}_{fname}.csv"
            out.to_csv(out_path, index=False)
            print(f"  [dim]Saved to[/dim] {out_path}")
            
from typing import List, Tuple
import pandas as pd
import os


def summarize_model_returns(input_dir: str = "../render_logs/model_test_results",
                            output_dir: str = "../render_logs/model_test_summary") -> Tuple[dict, pd.DataFrame]:
    model_returns = {}
    os.makedirs(output_dir, exist_ok=True)

    for csv_file in os.listdir(input_dir):
        if csv_file.endswith(".csv") and "_summary" not in csv_file:
            full_path = os.path.join(input_dir, csv_file)
            df = pd.read_csv(full_path)

            model_name = csv_file.split("_")[0]
            filename = "_".join(csv_file.split("_")[1:]).replace(".csv", "")
            final_return = df["cumulative_return"].iloc[-1]

            if model_name not in model_returns:
                model_returns[model_name] = []

            model_returns[model_name].append((filename, final_return))

    summary_frames = []
    for model_name, records in model_returns.items():
        df_model = pd.DataFrame(records, columns=["filename", "return"])
        df_model.sort_values("return", ascending=False, inplace=True)

        model_file_path = os.path.join(output_dir, f"{model_name}_summary.csv")
        df_model.to_csv(model_file_path, index=False)

        avg_return = df_model["return"].mean()
        summary_frames.append((model_name, avg_return))

    df_summary = pd.DataFrame(summary_frames, columns=["model", "avg_return"])
    df_summary.sort_values("avg_return", ascending=False, inplace=True)
    comparison_path = os.path.join(output_dir, "model_comparison_summary.csv")
    df_summary.to_csv(comparison_path, index=False)

    return model_returns, df_summary





if __name__ == "__main__":
    main()
    summarize_model_returns()