import numpy as np
from collections import defaultdict
from collections import deque


emotion_counts = defaultdict(int)
neutral_counter = 0
max_euphoria = 0

def reward_euphoric_reckless(history):
    global emotion_counts, neutral_counter, max_euphoria
    
    try:
        price_now = history["data_close", -1]
        price_prev = history["data_close", -2]
        position_now = history["position", -1]  # -1, -0.5, 0, 0.5, 1
        position_prev = history["position", -2]
        portfolio_now = history["portfolio_valuation", -1]
        portfolio_prev = history["portfolio_valuation", -2]
        time_now = history["date", -1]
        time_prev = history["date", -2]
    except (KeyError, IndexError):
        return 0.0

    timeframe_min = (time_now - time_prev) / np.timedelta64(1, 'm')
    scale_factor = np.sqrt(max(timeframe_min, 1) / 5) 

    fomo_thresh = 0.005 * scale_factor       # 0.5% for 5m, ~1.7% for 1h
    euphoria_thresh = 0.003 * scale_factor   # 0.3% for 5m, ~1% for 1h
    drawdown_thresh = -0.015 * scale_factor  # -1.5% for 5m, ~-5% for 1h

    price_change = (price_now - price_prev) / max(price_prev, 1e-8)
    portfolio_return = (portfolio_now - portfolio_prev) / max(portfolio_prev, 1e-8)
    abs_position = abs(position_now)

    reward = portfolio_return * 15.0  # 15x leverage mentality

    # EUPHORIA TRIGGER
    if position_prev != 0:
        is_win = (position_prev * price_change) > euphoria_thresh
        if is_win:
            win_streak = emotion_counts.get('win_streak', 0) + 1
            emotion_counts['win_streak'] = win_streak
            reward += 2.0 * win_streak * abs(position_prev)  # Streak bonus
            emotion_counts['euphoria'] += 1
        else:
            emotion_counts['win_streak'] = 0

    # FOMO PENALTY 
    if position_now == 0 and abs(price_change) > fomo_thresh:
        reward -= 4.0  
        emotion_counts['regret'] += 1

    if position_prev == position_now and portfolio_return < drawdown_thresh:
        reward += 3.0 * abs_position  # Reward conviction
        emotion_counts['inaction_in'] += 1

    # POSITION COMMITMENT
    reward += 0.3 * (position_now ** 2)  # Quadratic reward for full positions

    # NEUTRALITY PENALTY (escalating)
    if position_now == 0:
        neutral_counter += 1
        reward -= neutral_counter * 0.7  # Ramp punishment
        emotion_counts['neutral'] += 1
    else:
        neutral_counter = 0

    # VOLATILITY BONUS
    reward += np.tanh(abs(price_change) * 10) * 2.0  # Saturation for huge moves

    return float(reward)




def reward_confident_aware(history):
    global emotion_counts
    emotion_counts = emotion_counts if "emotion_counts" in globals() else {
        "euphoria": 0, "panic": 0, "regret": 0, "inaction": 0, "neutral": 0
    }

    try:
        price_now = history["data_close", -1]
        price_prev = history["data_close", -2]
        position_now = history["position", -1]
        position_prev = history["position", -2]
        portfolio_now = history["portfolio_valuation", -1]
        portfolio_prev = history["portfolio_valuation", -2]
        time_now = history["date", -1]
        time_prev = history["date", -2]
    except KeyError:
        return 0.0

    timeframe_minutes = (time_now - time_prev) / np.timedelta64(1, 'm')
    price_change = (price_now - price_prev) / max(price_prev, 1e-8)

    # --- Scaled thresholds ---
    up_thresh = 0.00004 * timeframe_minutes
    big_move = 0.0002 * timeframe_minutes
    extreme_move = 0.0007 * timeframe_minutes

    # --- Portfolio performance ---
    portfolio_change = (portfolio_now - portfolio_prev) / max(portfolio_prev, 1e-8)
    reward = 5 * portfolio_change

    # --- Emotion tagging ---
    if position_now > 0 and price_change > big_move:
        reward += 0.2
        emotion_counts["euphoria"] += 1

    elif position_now < 0 and price_change < -big_move:
        reward += 0.2
        emotion_counts["euphoria"] += 1

    elif position_now != 0 and abs(price_change) < 0.00002 * timeframe_minutes:
        reward -= 0.1
        emotion_counts["regret"] += 1

    elif position_prev == 1 and position_now == 0 and price_change < -big_move:
        emotion_counts["panic"] += 1

    elif position_prev == 1 and position_now == 0 and price_change > big_move:
        emotion_counts["regret"] += 1

    elif position_prev == 0 and position_now == 0 and abs(price_change) > extreme_move:
        emotion_counts["inaction"] += 1

    elif position_now == 0 and price_change > big_move:
        reward -= 0.2
        emotion_counts["regret"] += 1

    elif abs(position_now) == 1:
        emotion_counts["neutral_in"] += 1
    else:
        emotion_counts["neutral_out"] += 1

    if position_now == position_prev and abs(price_change) > big_move:
        reward += 0.15  # Holding into a move

    if position_now != position_prev:
        reward -= 0.02

    if position_now == 0:
        reward -= 0.05

    reward += 0.01 * abs(position_now)

    return float(reward)


def emotion_reward_function(history):
    return