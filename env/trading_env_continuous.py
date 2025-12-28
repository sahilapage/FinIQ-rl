import gym
import numpy as np
import torch
from gym import spaces


class ContinuousTradingEnv(gym.Env):
    def __init__(
        self,
        windows_tensor,
        encoder,
        initial_balance=10000,
        transaction_cost=0.0005,
        lambda_dd=0.001,
    ):
        super().__init__()

        self.windows = windows_tensor
        self.encoder = encoder

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_value = initial_balance

        self.transaction_cost = transaction_cost
        self.lambda_dd = lambda_dd

        self.current_step = 0

        # Continuous position [0,1]
        self.position = 0.0
        self.prev_position = 0.0

        # Action = desired position
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.state_dim = 128
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.peak_value = self.initial_balance

        self.position = 0.0
        self.prev_position = 0.0

        return self._get_state(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if self.current_step >= len(self.windows) - 2:
            return self._get_state(), 0.0, True, False, {}

        # Clip & disable shorting for now
        desired_position = float(np.clip(action[0], 0.0, 1.0))

        curr_price = self._get_price(self.current_step)
        next_price = self._get_price(self.current_step + 1)
        price_change = next_price - curr_price

        # Transaction cost on position change
        position_change = abs(desired_position - self.position)
        reward -= self.transaction_cost * position_change

        # Update position
        self.prev_position = self.position
        self.position = desired_position

        # PnL
        pnl = self.position * price_change
        reward += pnl
        self.balance += pnl

        # Update peak & drawdown
        self.peak_value = max(self.peak_value, self.balance)
        drawdown = max(0.0, self.peak_value - self.balance)
        reward -= self.lambda_dd * drawdown

        # Step forward
        self.current_step += 1

        if self.current_step >= len(self.windows) - 2:
            terminated = True

        obs = self._get_state()

        info = {
            "balance": self.balance,
            "position": self.position,
            "drawdown": drawdown,
        }

        return obs, reward, terminated, truncated, info

    def _get_state(self):
        window = self.windows[self.current_step].unsqueeze(0)
        with torch.no_grad():
            state = self.encoder(window)
        return state.squeeze(0).cpu().numpy()

    def _get_price(self, step):
        return self.windows[step][3, -1].item()
