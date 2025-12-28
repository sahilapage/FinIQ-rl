import torch
import numpy as np
import os
from stable_baselines3 import PPO

from env.trading_env_continuous import ContinuousTradingEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

DATA_DIR = "data/test"
MODEL_PATH = "ppo_phase5_continuous_multi"

model = PPO.load(MODEL_PATH)

results = {}

for file in os.listdir(DATA_DIR):
    if not file.endswith("_test.pt"):
        continue

    asset = file.replace("_test.pt", "")
    windows = torch.load(os.path.join(DATA_DIR, file))

    num_features = windows.shape[1]

    cnn = MarketCNN(in_channels=num_features)
    lstm = MarketLSTM(input_dim=num_features)
    encoder = MarketEncoder(cnn, lstm)

    for p in encoder.parameters():
        p.requires_grad = False

    env = ContinuousTradingEnv(windows, encoder)
    obs, _ = env.reset()

    positions, balances = [], []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        positions.append(info["position"])
        balances.append(info["balance"])

    results[asset] = {
        "final_balance": balances[-1],
        "avg_position": float(np.mean(positions)),
        "max_position": float(np.max(positions)),
    }

# ======================
# PRINT RESULTS
# ======================
print("\n===== PHASE 5 MULTI-ASSET RESULTS =====")
for asset, r in results.items():
    print(f"{asset}")
    print(f"  Final Balance : {r['final_balance']:.2f}")
    print(f"  Avg Position  : {r['avg_position']:.3f}")
    print(f"  Max Position  : {r['max_position']:.3f}")
    print("-----------------------------------")
