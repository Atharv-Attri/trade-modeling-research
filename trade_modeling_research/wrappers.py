import gymnasium as gym
from gymnasium import Env

def correct_valorisation(self, price):
    borrowed_asset = max(-self.asset, 0)
    long_asset = max(self.asset, 0)
    return (
        self.fiat
        + long_asset * price
        - borrowed_asset * price
        - self.interest_asset * price
        - self.interest_fiat
    )

class StableEnvWrapper(gym.Wrapper):

    def __init__(self, env: Env):
        super().__init__(env)
        self._already_done = False
        self._patched = False
        self._initial_val = None

    def reset(self, **kwargs):
        self._already_done = False
        obs, info = self.env.reset(**kwargs)

        price0 = self.unwrapped._price_array[self.unwrapped._idx]
        port_val = self.unwrapped._portfolio.valorisation(price0)
        #print(f"[reset] idx={self.unwrapped._idx} | price0={price0:.4f} | V0={port_val:.4f}")

        self._initial_val = port_val
        info["cumulative_return"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

    
        self._already_done = done or truncated

        # === Get current portfolio valuation ===
        price_now = self.unwrapped._price_array[self.unwrapped._idx]
        port_val  = self.unwrapped._portfolio.valorisation(price_now)

        info["portfolio_valuation"] = port_val

        # === Compute cumulative return anchored to V₀ ===
        if port_val is not None and self._initial_val is not None:
            cum_ret = (port_val / self._initial_val) - 1.0
            info["cumulative_return"] = cum_ret
            #print(
            #    f"[step] idx={self.unwrapped.current_index} | price={price_now:.2f} | val={port_val:.4f} | "
            #    f"V0={self._initial_val:.4f} | return={cum_ret:.6f}"
            #)
        else:
            info["cumulative_return"] = 0.0
            print("[step] Missing valuation for cumulative return")

        return obs, reward, done, truncated, info
