import random
import gym

from env.trading_env import TradingEnv


class MultiAssetTradingEnv(gym.Env):
    def __init__(self, asset_windows_dict, encoder, lambda_dd=0.001):
        super().__init__()

        self.asset_windows = asset_windows_dict
        self.encoder = encoder
        self.lambda_dd = lambda_dd
        self.asset_names = list(asset_windows_dict.keys())

        # Build a dummy env to expose spaces
        sample_asset = self.asset_names[0]
        dummy_env = TradingEnv(
            windows_tensor=self.asset_windows[sample_asset],
            encoder=self.encoder,
            lambda_dd=self.lambda_dd
        )

        self.action_space = dummy_env.action_space
        self.observation_space = dummy_env.observation_space

        self.current_env = None

    def seed(self, seed=None):
        random.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        asset = random.choice(self.asset_names)

        self.current_env = TradingEnv(
            windows_tensor=self.asset_windows[asset],
            encoder=self.encoder,
            lambda_dd=self.lambda_dd
        )

        return self.current_env.reset(seed=seed)

    def step(self, action):
        return self.current_env.step(action)
