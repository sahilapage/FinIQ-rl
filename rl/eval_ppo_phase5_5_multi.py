import torch
import numpy as np
from stable_baselines3 import PPO

from env.multi_asset_vol_target_env import MultiAssetVolTargetEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

# -----------------------------
# Load trained model
# -----------------------------
MODEL_NAME = "ppo_phase5_5_multi"
model = PPO.load(MODEL_NAME)

# -----------------------------
# Load test assets
# -----------------------------
asset_names = ["AAPL", "MSFT", "NVDA", "TSLA"]
asset_tensors = {}

for name in asset_names:
    asset_tensors[name] = torch.load(f"data/test/{name}_test.pt")

num_features = asset_tensors[asset_names[0]].shape[1]

cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

print("\n===== PHASE 5.5 MULTI-ASSET EVALUATION =====\n")

# ===============================
# Evaluate each asset separately
# ===============================
for asset in asset_names:
    env = MultiAssetVolTargetEnv(
        asset_tensors={asset: asset_tensors[asset]},
        encoder=encoder,
    )

    obs, _ = env.reset()

    positions = []
    balances = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        positions.append(info["position"])
        balances.append(info["balance"])

    positions = np.array(positions)

    final_balance = balances[-1]
    pnl = final_balance - env.initial_balance

    print(f"Asset: {asset}")
    print(f"  Final Balance : {final_balance:.2f}")
    print(f"  Total PnL     : {pnl:.4f}")
    print(f"  Avg Position  : {positions.mean():.3f}")
    print(f"  Max Position  : {positions.max():.3f}")
    print(f"  Std Position  : {positions.std():.3f}")
    print("------------------------------------------")
