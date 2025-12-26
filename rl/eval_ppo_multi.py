import os
import argparse
import torch
import numpy as np
import random

from stable_baselines3 import PPO

from env.trading_env import TradingEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

# ===============================
# CLI ARGUMENTS
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to trained PPO model (e.g. ppo_lambda_0.001)"
)
args = parser.parse_args()

MODEL_PATH = args.model

# ===============================
# Reproducibility
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===============================
# Load PPO model
# ===============================
model = PPO.load(MODEL_PATH)

# ===============================
# Load test assets
# ===============================
DATA_DIR = "data/test"
asset_files = sorted(os.listdir(DATA_DIR))

# ===============================
# Build encoder (same as training)
# ===============================
sample_tensor = torch.load(os.path.join(DATA_DIR, asset_files[0]))
num_features = sample_tensor.shape[1]

cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)
encoder.eval()

# ===============================
# Evaluation
# ===============================
print(f"\n===== MULTI-ASSET EVALUATION | MODEL: {MODEL_PATH} =====\n")

results = []

for file in asset_files:
    asset_name = file.replace("_test.pt", "")
    windows_tensor = torch.load(os.path.join(DATA_DIR, file))

    env = TradingEnv(
        windows_tensor=windows_tensor,
        encoder=encoder
    )

    obs, _ = env.reset()
    done = False

    total_reward = 0.0
    executed_trades = 0
    unique_actions = set()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        unique_actions.add(int(action))

        prev_position = env.positions

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward

        # ----- EXECUTION-BASED TRADE COUNT -----
        if prev_position == 0 and env.positions == 1:
            executed_trades += 1   # BUY
        elif prev_position == 1 and env.positions == 0:
            executed_trades += 1   # SELL

    final_balance = env.balance
    pnl = final_balance - env.initial_balance
    avg_reward = total_reward / max(1, env.current_step)

    print(f"Asset: {asset_name}")
    print(f"  Final Balance     : {final_balance:.2f}")
    print(f"  Total PnL         : {pnl:.4f}")
    print(f"  Executed Trades   : {executed_trades}")
    print(f"  Avg Reward        : {avg_reward:.6f}")
    print(f"  Unique Actions    : {unique_actions}")
    print("-" * 50)

    results.append((asset_name, pnl, executed_trades))

# ===============================
# Aggregate summary (IMPORTANT)
# ===============================
avg_pnl = np.mean([r[1] for r in results])
total_trades = sum(r[2] for r in results)

print("\n===== AGGREGATE SUMMARY =====")
print(f"Average PnL (4 assets): {avg_pnl:.4f}")
print(f"Total Executed Trades : {total_trades}")
print("=" * 50)
