import gym
import random
import numpy as np
from gym import spaces

from env.trading_env_continuous import ContinuousTradingEnv


class MultiAssetContinuousEnv(gym.Env):
    """
    A proper Gym environment that randomly samples one asset per episode.
    """

    metadata = {"render.modes": []}

    def __init__(self, asset_windows_dict, encoder, **env_kwargs):
        super().__init__()

        self.asset_windows_dict = asset_windows_dict
        self.encoder = encoder
        self.env_kwargs = env_kwargs

        self.current_asset = None
        self.env = None

        # We borrow action & observation spaces from a dummy env
        sample_windows = next(iter(asset_windows_dict.values()))
        dummy_env = ContinuousTradingEnv(
            windows_tensor=sample_windows,
            encoder=encoder,
            **env_kwargs
        )

        self.action_space = dummy_env.action_space
        self.observation_space = dummy_env.observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly choose asset each episode
        self.current_asset = random.choice(list(self.asset_windows_dict.keys()))
        windows = self.asset_windows_dict[self.current_asset]

        self.env = ContinuousTradingEnv(
            windows_tensor=windows,
            encoder=self.encoder,
            **self.env_kwargs
        )

        obs, info = self.env.reset(seed=seed)
        info["asset"] = self.current_asset
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["asset"] = self.current_asset
        return obs, reward, terminated, truncated, info
