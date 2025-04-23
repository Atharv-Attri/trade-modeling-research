from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import os
from rich import print

class MetricsLogger(BaseCallback):
    def __init__(
        self,
        log_path: str = "../render_logs/training_logs/",
        filename: str = "training_metrics.csv",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.log_path = log_path
        self.filename = filename
        self.metrics = []
        self._last_ev = None

        # Track episode rewards and lengths
        self._episode_rewards = []
        self._episode_lengths = []
        self._current_rewards = None
        self._current_lengths = None

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self._current_rewards = [0.0 for _ in range(n_envs)]
        self._current_lengths = [0 for _ in range(n_envs)]

    def _on_step(self) -> bool:
        ev = self.model.logger.name_to_value.get("train/explained_variance", None)

        # Step tracking for each env
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        for i, (r, d) in enumerate(zip(rewards, dones)):
            self._current_rewards[i] += r
            self._current_lengths[i] += 1

            if d:
                self._episode_rewards.append(self._current_rewards[i])
                self._episode_lengths.append(self._current_lengths[i])
                self._current_rewards[i] = 0.0
                self._current_lengths[i] = 0

        if ev is not None and ev != self._last_ev:
            self._last_ev = ev

            reward_mean = (
                sum(self._episode_rewards[-100:]) / len(self._episode_rewards[-100:])
                if self._episode_rewards else None
            )
            ep_len_mean = (
                sum(self._episode_lengths[-100:]) / len(self._episode_lengths[-100:])
                if self._episode_lengths else None
            )

            row = {
                "timesteps": self.num_timesteps,
                "explained_variance": ev,
                "approx_kl": self.model.logger.name_to_value.get("train/approx_kl"),
                "value_loss": self.model.logger.name_to_value.get("train/value_loss"),
                "policy_gradient_loss": self.model.logger.name_to_value.get("train/policy_gradient_loss"),
                "entropy_loss": self.model.logger.name_to_value.get("train/entropy_loss"),
                "clip_fraction": self.model.logger.name_to_value.get("train/clip_fraction"),
                "learning_rate": self.model.logger.name_to_value.get("train/learning_rate"),
                "reward_mean": reward_mean,
                "episode_len": ep_len_mean,
            }

            self.metrics.append(row)

            if self.verbose > 0:
                print(
                    f"[bold cyan]{self.num_timesteps:,}[/] "
                    f"[white]EV:[/] [green]{ev:.3f}[/]  "
                    f"[white]KL:[/] [yellow]{row['approx_kl']:.5f}[/]  "
                    f"[white]Reward:[/] [magenta]{reward_mean:.2f}[/]  "
                    f"[white]Len:[/] [blue]{ep_len_mean:.1f}[/]"
                )

        return True

    def _on_training_end(self) -> None:
        os.makedirs(self.log_path, exist_ok=True)
        df = pd.DataFrame(self.metrics)
        path = os.path.join(self.log_path, self.filename)
        df.to_csv(path, index=False)
        if self.verbose > 0:
            print(f"[bold green][MetricsLogger][/bold green] Saved metrics to [underline]{path}[/]")
