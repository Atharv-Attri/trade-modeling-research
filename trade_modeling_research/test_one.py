import pandas as pd
from data import get_stock_data
from datetime import datetime, timedelta
import gymnasium as gym
import gym_trading_env
import numpy as np



pd.set_option('future.no_silent_downcasting', True)
symbol = "PLTR"
timeframe = "1h"
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

print(f"Fetching {symbol} data for the past 3 days with a {timeframe} timeframe (extended hours)...")
#df = get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date, tf=timeframe, extended=True, cached=False)

df = pd.read_csv("../data/PLTR_2025-02-09_2025-04-10_1h_ext.csv", parse_dates=["datetime"])
print(df.index)
df.set_index("datetime", inplace=True)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)
df.sort_index(inplace=True)
df.index = df.index.tz_localize(None)

# 1. Relative close change: (close[t] - close[t-1]) / close[t-1]
df["feature_close"] = df["close"].pct_change()

# 2. Open to close ratio
df["feature_open"] = df["open"] / df["close"]

# 3. High to close ratio
df["feature_high"] = df["high"] / df["close"]

# 4. Low to close ratio
df["feature_low"] = df["low"] / df["close"]

# 5. Volume to max volume over the past 7 days (rolling 7*24 hours for 1h bars)
df["feature_volume"] = df["volume"]

# 6. Final cleanup
df.dropna(inplace=True)

env = gym.make("TradingEnv",
        name= "PLTR",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.00/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.000/100, # 0.0003% per timestep (one timestep = 1h here)
        initial_position=0,

    )

done, truncated = False, False
observation, info = env.reset()
while not done and not truncated:
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
    position_index = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
    observation, reward, done, truncated, info = env.step(position_index)
    env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
    env.add_metric('Episode Length', lambda history : len(history['position']) )

env.save_for_render(dir = "../render_logs")


print(df.head())
