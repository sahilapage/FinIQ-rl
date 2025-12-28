import gym
import numpy as np
import torch
from gym import spaces
import random

class MultiAssetVolTargetEnv(gym.Env):
    def __init__(
        self,
        asset_tensors: dict,
        encoder,
        initial_balance=10000,
        transaction_cost=0.0005,
        vol_lookback=20,
    ):
        super().__init__()

        self.asset_tensors = asset_tensors
        self.encoder = encoder
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.vol_lookback = vol_lookback

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.state_dim = 128
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self._sample_asset()

    # -------------------------
    def _sample_asset(self):
        self.asset_name = random.choice(list(self.asset_tensors.keys()))
        self.windows = self.asset_tensors[self.asset_name]
        self.current_step = self.vol_lookback
        self.balance = self.initial_balance
        self.position = 0.0

    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_asset()
        return self._get_state(), {}

    # -------------------------
    def step(self, action):
        terminated = False
        truncated = False

        action = float(np.clip(action[0], -1.0, 1.0))

        curr_price = self._get_price(self.current_step)
        next_price = self._get_price(self.current_step + 1)
        price_change = next_price - curr_price

        volatility = self._compute_volatility()
        volatility = np.clip(volatility, 0.005, 0.05)

        pnl = self.position * price_change
        reward = pnl / volatility

        alpha = 0.05 / volatility
        reward -= alpha * (self.position ** 2)

        signal_bonus = 0.002 * abs(self.position)
        reward += signal_bonus

        reward -= self.transaction_cost * abs(action - self.position)

        self.position = action
        self.balance += pnl

        self.current_step += 1
        if self.current_step >= len(self.windows) - 2:
            terminated = True

        info = {
            "asset": self.asset_name,
            "balance": self.balance,
            "position": self.position,
            "volatility": volatility,
        }

        return self._get_state(), reward, terminated, truncated, info

    # -------------------------
    def _get_state(self):
        window = self.windows[self.current_step].unsqueeze(0)
        with torch.no_grad():
            state = self.encoder(window)
        return state.squeeze(0).cpu().numpy()

    def _get_price(self, step):
        return self.windows[step][3, -1].item()

    def _compute_volatility(self):
        prices = self.windows[
            self.current_step - self.vol_lookback : self.current_step + 1,
            3,
            -1
        ].cpu().numpy()

        prices = np.clip(prices, 1e-6, None)
        returns = np.diff(np.log(prices))
        return np.std(returns)
