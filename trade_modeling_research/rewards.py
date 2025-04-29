import numpy as np
import random
from collections import defaultdict
from ordered_trading_env import OrderedMultiDatasetTradingEnv
from wrappers import StableEnvWrapper

# --- Shared Counters ---
emotion_counts = defaultdict(int)
neutral_counter = 0

# --- Reward Functions ---

def reward_euphoric_reckless(history):
    """
    YOLO trader: high leverage, euphoria on big wins (long or short),
    punishes FOMO, overlong, missed shorts.
    """
    global emotion_counts, neutral_counter

    try:
        price_now    = history["data_close", -1]
        price_prev   = history["data_close", -2]
        position_now = history["position",   -1]
        position_prev= history["position",   -2]
        port_now     = history["portfolio_valuation", -1]
        port_prev    = history["portfolio_valuation", -2]
        t_now        = history["date", -1]
        t_prev       = history["date", -2]
    except (KeyError, IndexError):
        return 0.0

    # time & scale
    dt_min       = (t_now - t_prev) / np.timedelta64(1, 'm')
    scale        = np.sqrt(max(dt_min, 1) / 5)

    # thresholds
    euphoric_th  = 0.003 * scale
    fomo_th      = 0.005 * scale
    drawdown_th  = -0.015 * scale

    price_chg    = (price_now - price_prev) / max(price_prev, 1e-8)
    port_ret     = (port_now - port_prev) / max(port_prev, 1e-8)
    abs_pos      = abs(position_now)

    reward = port_ret * 15.0

    # EUPHORIA (long or short wins)
    if position_prev != 0:
        win = (position_prev * price_chg) > euphoric_th
        if win:
            streak = emotion_counts.get('win_streak', 0) + 1
            emotion_counts['win_streak'] = streak
            reward += 2.0 * streak * abs(position_prev)
            emotion_counts['euphoria'] += 1
        else:
            emotion_counts['win_streak'] = 0

    # YOLO SHORT BONUS (big down moves)
    if position_now < 0 and price_chg < -euphoric_th:
        reward += 2.5 * abs_pos
        emotion_counts['euphoria'] += 1
        emotion_counts['short_win'] += 1

    # FOMO penalty (flat when big move)
    if position_now == 0 and abs(price_chg) > fomo_th:
        reward -= 4.0
        emotion_counts['regret'] += 1

    # conviction reward
    if position_prev == position_now and port_ret < drawdown_th:
        reward += 3.0 * abs_pos
        emotion_counts['inaction_in'] += 1

    # quadratic position bonus
    reward += 0.3 * (position_now ** 2)

    # missed short penalty
    if position_now == 0 and price_chg < -fomo_th:
        reward -= 0.5
        emotion_counts['missed_short'] += 1

    # neutral penalty ramp
    if position_now == 0:
        neutral_counter += 1
        reward -= neutral_counter * 0.7
        emotion_counts['neutral'] += 1
    else:
        neutral_counter = 0

    # volatility bonus
    reward += np.tanh(abs(price_chg) * 10) * 2.0

    # overlong penalty
    if position_now == 1.0:
        emotion_counts['overlong'] += 1
        reward -= 0.3 * min(emotion_counts['overlong'], 5)
    else:
        emotion_counts['overlong'] = 0

    return float(reward)


def reward_confident_aware(history):
    """
    Confident trader: scaled portfolio returns, rewards well-timed longs & shorts,
    penalizes indecision and inaction.
    """
    global emotion_counts

    try:
        price_now    = history["data_close", -1]
        price_prev   = history["data_close", -2]
        position_now = history["position",   -1]
        position_prev= history["position",   -2]
        port_now     = history["portfolio_valuation", -1]
        port_prev    = history["portfolio_valuation", -2]
        t_now        = history["date", -1]
        t_prev       = history["date", -2]
    except (KeyError, IndexError):
        return 0.0

    dt_min    = (t_now - t_prev) / np.timedelta64(1, 'm')
    price_chg = (price_now - price_prev) / max(price_prev, 1e-8)
    port_chg  = (port_now - port_prev) / max(port_prev, 1e-8)

    reward = 5.0 * port_chg

    # thresholds
    big_move      = 0.0002 * dt_min
    mild_vol      = 0.00004 * dt_min
    sharp_move    = 0.0007 * dt_min

    # LONG euphoria
    if position_now > 0 and price_chg > big_move:
        reward += 0.5
        emotion_counts['euphoria'] += 1

    # SHORT bonus
    if position_now < 0 and price_chg < -big_move:
        reward += 0.5
        emotion_counts['euphoria'] += 1
        emotion_counts['short_win'] += 1

    # bored regret
    if position_now != 0 and abs(price_chg) < mild_vol:
        reward -= 0.1
        emotion_counts['regret'] += 1

    # panic/regret on exit
    if position_prev == 1 and position_now == 0:
        if price_chg < -big_move:
            emotion_counts['panic'] += 1
        elif price_chg > big_move:
            emotion_counts['regret'] += 1

    # missed short entry
    if position_now == 0 and price_chg < -big_move:
        reward -= 0.3
        emotion_counts['missed_short'] += 1

    # confidence hold bonus
    if position_now == position_prev and abs(price_chg) > big_move:
        reward += 0.25

    # action switch penalty
    if position_now != position_prev:
        reward -= 0.02

    # neutral penalty
    if position_now == 0:
        reward -= 0.05
        emotion_counts['neutral'] += 1

    # exploration bonus for taking partial
    if abs(position_now) == 0.5:
        reward += 0.1

    # overlong penalty
    if position_now == 1.0:
        emotion_counts['overlong'] += 1
        reward -= 0.3 * min(emotion_counts['overlong'], 5)
    else:
        emotion_counts['overlong'] = 0

    return float(reward)


def reward_risk_averse_mindful(history):
    """
    Cautious defender: penalizes overhold, rewards protective shorts,
    moderate conviction and clean exits.
    """
    global emotion_counts, neutral_counter

    try:
        price_now    = history["data_close", -1]
        price_prev   = history["data_close", -2]
        position_now = history["position",   -1]
        position_prev= history["position",   -2]
        port_now     = history["portfolio_valuation", -1]
        port_prev    = history["portfolio_valuation", -2]
        t_now        = history["date", -1]
        t_prev       = history["date", -2]
    except (KeyError, IndexError):
        return 0.0

    # position duration
    duration = 1
    for i in range(2, len(history["position"])):
        if history["position", -i] == position_now:
            duration += 1
        else:
            break

    dt_min        = (t_now - t_prev) / np.timedelta64(1, 'm')
    price_chg     = (price_now - price_prev) / max(price_prev, 1e-8)
    port_chg      = (port_now - port_prev) / max(port_prev, 1e-8)
    abs_pos       = abs(position_now)

    reward = 1.5 * port_chg

    # sharpening thresholds
    mild_vol      = 0.0025 * dt_min
    sharp_loss    = -0.004 * dt_min
    profit_tp     = 0.0025

    # protective short reward
    if position_now < 0 and price_chg < -mild_vol:
        reward += 0.4 * abs_pos
        emotion_counts['short_win'] += 1

    # sharp drop defense
    if position_prev == 0 and position_now < 0 and price_chg < sharp_loss:
        reward += 0.5
        emotion_counts['euphoria'] += 1

    # overhold penalty
    pos_pnl = position_now * price_chg
    if pos_pnl < -0.002 * dt_min:
        reward -= 0.6 * abs(pos_pnl) * min(duration, 5)
        emotion_counts['regret-overhold'] += 1
        if abs_pos == 1:
            reward -= 0.4
            emotion_counts['regret-overexposed'] += 1

    # conviction rewards
    if position_now == position_prev and abs_pos > 0 and pos_pnl > 0:
        reward += 0.35 * min(duration, 4)
        emotion_counts['euphoria'] += 1

    # smart exit
    if position_prev != 0 and position_now == 0:
        if port_chg > profit_tp:
            reward += 0.8 * port_chg
            emotion_counts['euphoria-takeprofit'] += 1
        elif port_chg < -profit_tp:
            reward -= 0.5 * abs(port_chg)
            emotion_counts['regret-lateexit'] += 1

    # missed short penalty
    if position_now == 0 and price_chg < -mild_vol:
        reward -= 0.5
        emotion_counts['missed_short'] += 1

    # neutral penalty
    if position_now == 0:
        neutral_counter += 1
        if neutral_counter > 4:
            reward -= 0.1 * (neutral_counter - 4)
            emotion_counts['inaction-out'] += 1
    else:
        neutral_counter = 0

    # overlong penalty
    if position_now == 1.0:
        emotion_counts['overlong'] += 1
        reward -= 0.4 * min(emotion_counts['overlong'], 5)
    else:
        emotion_counts['overlong'] = 0

    return float(reward)


