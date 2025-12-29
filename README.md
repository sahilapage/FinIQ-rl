# RegimeRL
## Regime-Aware Reinforcement Learning for Multi-Asset Trading

RegimeRL is a research-oriented reinforcement learning framework for algorithmic trading that focuses on decision-making under market regimes, rather than price prediction.

The system learns position sizing and trade timing directly from market data using deep reinforcement learning, with explicit modeling of risk, drawdowns, transaction costs, and regime dynamics.

## Core Idea

Instead of predicting future prices, RegimeRL learns a policy that answers:

“Given the current market state and regime, what position should be held?”

This formulation aligns more closely with real-world quantitative trading systems.

## Architecture Overview
```ruby

OHLCV Windows
     ↓
1D CNN (local market patterns)
     ↓
LSTM (temporal dependencies)
     ↓
Latent Market State (128-D)
     ↓
PPO Policy
     ↓
Continuous Position Size

```

## Key Components

### Market Representation

• CNN extracts short-term price and volume structure
• LSTM captures temporal dynamics
• Produces a compact latent state used by the RL agent

### Trading Environment

• Custom Gym-style environment with:
• Transaction costs
• Realized & unrealized PnL
• Drawdown-aware penalties
• Minimum holding constraints
• Market regime detection (sideways / volatile / trending)

### Reinforcement Learning

1. Proximal Policy Optimization (PPO)
2. Continuous action space (position sizing)
3. Stable training under non-stationary market data

### Regime Awareness

1. Market regimes detected online using rolling volatility and returns
2. Risk penalties adapt dynamically based on regime
3. Encourages cautious behavior in volatile markets and participation in trends

### Training & Evaluation

• Strict train/test separation
• Walk-forward and multi-asset evaluation
• No look-ahead bias
• No supervised labels
• Evaluation focuses on:
• Stability of positions
• Robustness across assets
• Risk-adjusted behavior (not peak returns)

## Project Structure
```ruby

RegimeRL/
├── env/
│   └── finiq_env.py        # Unified trading environment
├── models/
│   ├── cnn.py
│   ├── lstm.py
│   └── encoder.py
├── rl/
│   ├── train_finiq.py
│   └── eval_finiq.py
├── datasets/
├── README.md
└── .gitignore

```
## Usage

### Train
```ruby
python -m rl.train_finiq
```

### Evaluate
```ruby
python -m rl.eval_finiq
```

## Disclaimer

This project is for research and educational purposes only.
It does not constitute financial advice and is not intended for live trading.
