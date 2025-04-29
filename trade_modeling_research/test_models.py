import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ordered_trading_env import OrderedMultiDatasetTradingEnv
from wrappers import StableEnvWrapper
from rich import print
from typing import List, Tuple


# --- Preprocessing Function ---
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index("datetime", inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)

    if "close" not in df.columns and "data_close" in df.columns:
        df["close"] = df["data_close"]
    if "close" not in df.columns:
        raise ValueError("'close' column is required in data")

    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df.dropna(inplace=True)
    return df


# --- Environment Builder ---
def make_env_for_file(filepath: str):
    env = OrderedMultiDatasetTradingEnv(
        dataset_dir=filepath,
        preprocess=preprocess,
        positions=[-1.0, -0.5, 0.0, 0.5, 1.0],
        trading_fees=0.0,
        borrow_interest_rate=0.0,
        initial_position=0,
        max_episode_duration='max',
        shuffle_datasets=False,
        random_start=False,
        reward_function=lambda *args, **kwargs: 0.0,
        verbose=0,
    )
    return StableEnvWrapper(env)


# --- Model Evaluation ---
def evaluate_models_on_files(files: List[str], model_paths: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    positions = [-1.0, -0.5, 0.0, 0.5, 1.0]

    for name, model_path in model_paths.items():
        print(f"\n[bold cyan]Evaluating model:[/bold cyan] {name}")
        model = PPO.load(model_path)

        for file_path in files:
            fname = Path(file_path).stem
            print(f"[blue] → File:[/blue] {fname}")

            df_raw = pd.read_pickle(file_path)
            df_proc = preprocess(df_raw).reset_index()

            env = make_env_for_file(file_path)
            obs, info = env.reset()
            V0 = info.get("portfolio_valuation", 1.0)

            records = [
                {
                    "step": 0,
                    "open": df_proc.iloc[0]["open"],
                    "high": df_proc.iloc[0]["high"],
                    "low": df_proc.iloc[0]["low"],
                    "close": df_proc.iloc[0]["close"],
                    "action": 1.0,
                    "cumulative_return": 0.0,
                }
            ]

            done, truncated = False, False
            step = 1
            while not (done or truncated) and step < len(df_proc):
                action, _ = model.predict(np.expand_dims(obs, axis=0), deterministic=True)
                obs, reward, done, truncated, info = env.step(action[0])

                Vt = info.get("portfolio_valuation", V0)
                cum_frac = (Vt / V0) - 1.0

                row = df_proc.iloc[step]
                records.append({
                    "step": step,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "action": positions[action[0]],
                    "cumulative_return": cum_frac,
                })
                step += 1

            final_valuation = info.get("portfolio_valuation", V0)
            total_frac = (final_valuation / V0) - 1.0
            print(f"  [green]Total return:[/green] {total_frac:.6f} ({total_frac*100:.2f}%)")

            out = pd.DataFrame.from_records(records)
            out_path = f"{output_dir}/{name}_{fname}.csv"
            out.to_csv(out_path, index=False)
            print(f"  [dim]Saved to[/dim] {out_path}")


# --- Result Summary ---
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

            model_returns.setdefault(model_name, []).append((filename, final_return))

    summary_frames = []
    for model_name, records in model_returns.items():
        df_model = pd.DataFrame(records, columns=["filename", "return"])
        df_model.sort_values("return", ascending=False, inplace=True)
        df_model.to_csv(os.path.join(output_dir, f"{model_name}_summary.csv"), index=False)

        avg_return = df_model["return"].mean()
        summary_frames.append((model_name, avg_return))

    df_summary = pd.DataFrame(summary_frames, columns=["model", "avg_return"])
    df_summary.sort_values("avg_return", ascending=False, inplace=True)
    df_summary.to_csv(os.path.join(output_dir, "model_comparison_summary.csv"), index=False)

    return model_returns, df_summary


# --- Run on test_2025.csv ---
def evaluate_on_test_2025():
    df = pd.read_csv("../data/processed/test_2025.csv")
    tickers = df["ticker"].dropna().unique().tolist()
    test_dir = "../data/test_2025"

    all_files = []
    for ticker in tickers:
        matched_files = glob.glob(f"{test_dir}/{ticker}_*.pkl")
        all_files.extend(matched_files)

    model_paths = {
        "euphoric_reckless": "../models/ppo_reward_euphoric_reckless.zip",
        "confident_aware": "../models/ppo_reward_confident_aware.zip",
        "risk_averse_mindful": "../models/ppo_reward_risk_averse_mindful.zip",
    }
    evaluate_models_on_files(all_files, model_paths, output_dir="../render_logs/model_test_results")
    summarize_model_returns()


def summarize_model_action_counts(results_dir: str = "../render_logs/model_test_results"):
    action_summary = {}

    for file in os.listdir(results_dir):
        if not file.endswith(".csv"):
            continue

        model_name = file.split("_")[0]
        df = pd.read_csv(os.path.join(results_dir, file))

        # Count action frequency per file (treat each action in that file as 1 unit regardless of length)
        file_actions = set(df["action"].round(2))  # rounding to deal with float precision if needed

        if model_name not in action_summary:
            action_summary[model_name] = {}

        for action in file_actions:
            action_summary[model_name][action] = action_summary[model_name].get(action, 0) + 1

    # --- Print the final results ---
    for model, action_counts in action_summary.items():
        print(f"\n[bold blue]{model} action presence count across files:[/bold blue]")
        for action, count in sorted(action_counts.items()):
            print(f"  Action {action}: {count} files")


if __name__ == "__main__":
    #evaluate_on_test_2025()
    summarize_model_returns()
    summarize_model_action_counts()