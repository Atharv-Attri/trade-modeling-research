from gym_trading_env.renderer import Renderer
import pandas as pd
import numpy as np
#! IN PROGRESS DO NOT USE
renderer = Renderer(render_logs_dir="../render_logs")

# === ENTRY POINTS ===
def entry_points(df):
    # Detect transition from 0 to 1 (long) or 0 to -1 (short)
    change = df["position"].diff().fillna(0).astype(float)
    entries = df["close"].copy() * np.nan
    entries[change == 1] = df["close"][change == 1]  # Entry for long position
    entries[change == -1] = df["close"][change == -1]  # Entry for short position
    return entries

# === EXIT POINTS ===
def exit_points(df):
    # Detect transition from 1 to 0 (exit long) or -1 to 0 (exit short)
    exit_mask = (df["position"].shift(1) == 1) & (df["position"] == 0)  # Exit long
    exit_mask |= (df["position"].shift(1) == -1) & (df["position"] == 0)  # Exit short
    exits = df["close"].copy() * np.nan
    exits[exit_mask] = df["close"][exit_mask]
    return exits

# === Load your dataframe here ===
# Assuming you're loading a CSV or pickle, replace this line with actual loading code
df = pd.read_csv("../render_logs\output.csv")  # Adjust path as necessary

# === Add lines to chart ===
renderer.add_line(
    name="Entry",
    function=lambda df: entry_points(df),  # Pass df to entry_points
    line_options={"width": 2, "color": "green"}  # Green for entry points
)

renderer.add_line(
    name="Exit",
    function=lambda df: exit_points(df),  # Pass df to exit_points
    line_options={"width": 2, "color": "red"}  # Red for exit points
)

# === Add metrics if you want (optional) ===
renderer.add_metric(
    name="Total Trades",
    function=lambda df: f"{(df['position'].diff().fillna(0).astype(float) != 0).sum()}"
)

renderer.add_metric(
    name="Final Portfolio Value",
    function=lambda df: f"${df['portfolio_valuation'].iloc[-1]:,.2f}"
)

# === Run the renderer ===
renderer.run()
