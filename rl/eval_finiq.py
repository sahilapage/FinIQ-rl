import torch
import numpy as np
from stable_baselines3 import PPO

from env.finiq_env import FinIQEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

assets = ["AAPL", "MSFT", "NVDA", "TSLA"]
windows = {
    a: torch.load(f"data/test/{a}_test.pt")
    for a in assets
}

num_features = windows[assets[0]].shape[1]
cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

env = FinIQEnv(
    windows_dict=windows,
    encoder=encoder,
    action_type="continuous",
    regime_aware=True
)

model = PPO.load("ppo_finiq_final")

obs, _ = env.reset()
done = False

balances = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    balances.append(info["balance"])

print("\n===== FINAL RESULTS =====")
print("Final Balance:", balances[-1])
print("Total Return:", balances[-1] - balances[0])
