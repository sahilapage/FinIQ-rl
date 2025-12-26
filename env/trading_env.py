import gym
import numpy as np
import torch
from gym import spaces


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

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
        self.transaction_cost = transaction_cost
        self.lambda_dd = lambda_dd

        # Trading state
        self.current_step = 0
        self.balance = initial_balance
        self.peak_value = initial_balance
        self.positions = 0
        self.entry_price = 0.0

        # Structural fixes
        self.min_hold_steps = 10
        self.hold_counter = 0

        # Regime-specific risk aversion
        self.REGIME_LAMBDA = {
            "trending": 0.0003,
            "sideways": 0.001,
            "volatile": 0.003,
        }

        # Spaces
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        self.state_dim = 128
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

    # ---------------------------------------------------
    # RESET
    # ---------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.peak_value = self.initial_balance
        self.positions = 0
        self.entry_price = 0.0
        self.hold_counter = 0

        obs = self._get_state()
        return obs, {}

    # ---------------------------------------------------
    # STEP
    # ---------------------------------------------------
    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # Terminal guard
        if self.current_step >= len(self.windows) - 2:
            return self._get_state(), 0.0, True, False, {}

        # Enforce valid actions
        if not self._is_action_valid(action):
            action = 0  # HOLD

        curr_price = self._get_price(self.current_step)
        next_price = self._get_price(self.current_step + 1)

        # ---------------------------------------------------
        # ACTION LOGIC
        # ---------------------------------------------------
        # BUY
        if action == 1 and self.positions == 0:
            self.positions = 1
            self.entry_price = curr_price
            self.hold_counter = 0
            reward -= self.transaction_cost

        # SELL
        elif action == 2 and self.positions == 1:
            if self.hold_counter >= self.min_hold_steps:
                pnl = curr_price - self.entry_price
                self.balance += pnl
                reward += pnl
                reward -= self.transaction_cost
                self.positions = 0
                self.entry_price = 0.0
                self.hold_counter = 0
            else:
                action = 0  # force HOLD

        # HOLD
        if action == 0:
            reward -= 0.00001  # small inactivity drag

        # ---------------------------------------------------
        # POSITION DYNAMICS
        # ---------------------------------------------------
        if self.positions == 1:
            self.hold_counter += 1

            # Unrealized PnL
            unrealized = next_price - curr_price
            reward += unrealized

            # Update peak balance
            self.peak_value = max(self.peak_value, self.balance + unrealized)

            # Regime-aware risk penalty (ONLY when exposed)
            regime = self._detect_regime()
            lambda_dd = self.REGIME_LAMBDA.get(regime, self.lambda_dd)

            drawdown = max(0.0, self.peak_value - (self.balance + unrealized))
            reward -= lambda_dd * drawdown

            # Incentivize trend participation
            if regime == "trending":
                reward += 0.0005

        else:
            # Opportunity cost of staying in cash
            reward -= 0.00005

        # ---------------------------------------------------
        # STEP FORWARD
        # ---------------------------------------------------
        self.current_step += 1

        if self.current_step >= len(self.windows) - 2:
            terminated = True

        obs = self._get_state()
        info = {
            "balance": self.balance,
            "position": self.positions,
            "regime": self._detect_regime(),
        }

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------
    # STATE
    # ---------------------------------------------------
    def _get_state(self):
        window = self.windows[self.current_step].unsqueeze(0)
        with torch.no_grad():
            state = self.encoder(window)
        return state.squeeze(0).cpu().numpy()

    def _get_price(self, step):
        return self.windows[step][3, -1].item()

    # ---------------------------------------------------
    # REGIME DETECTION
    # ---------------------------------------------------
    def _detect_regime(self):
        lookback = 20

        if self.current_step < lookback + 1:
            return "sideways"

        close_prices = self.windows[
            self.current_step - lookback : self.current_step + 1,
            3,
            -1,
        ].cpu().numpy()

        close_prices = np.clip(close_prices, 1e-6, None)
        returns = np.diff(np.log(close_prices))

        mean_return = np.mean(returns)
        volatility = np.std(returns)

        if volatility > 0.015:
            return "volatile"
        elif abs(mean_return) > 0.002:
            return "trending"
        else:
            return "sideways"

    # ---------------------------------------------------
    # ACTION VALIDATION
    # ---------------------------------------------------
    def _is_action_valid(self, action):
        if self.positions == 0 and action == 2:
            return False
        if self.positions == 1 and action == 1:
            return False
        return True
