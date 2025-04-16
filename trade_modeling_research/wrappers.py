import gymnasium as gym
from gymnasium import Env

class StableEnvWrapper(gym.Wrapper):
    """
    Prevents stepping the environment after it's already done/truncated.
    Useful for compatibility with vectorized training or libraries like SB3.
    """
    def __init__(self, env: Env):
        super().__init__(env)
        self._already_done = False

    def reset(self, **kwargs):
        self._already_done = False
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._already_done:
            raise RuntimeError("Tried to step() after done=True or truncated=True. Call reset() first.")

        obs, reward, done, truncated, info = self.env.step(action)
        self._already_done = done or truncated
        return obs, reward, done, truncated, info
