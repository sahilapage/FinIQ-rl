import torch
import numpy as np
from stable_baselines3 import PPO

from env.trading_env_continuous import ContinuousTradingEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

# =========================
# LOAD DATA
# =========================
windows_tensor = torch.load("data/test/AAPL_test.pt")

num_features = windows_tensor.shape[1]

# =========================
# ENCODER
# =========================
cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

for p in encoder.parameters():
    p.requires_grad = False

# =========================
# ENV
# =========================
env = ContinuousTradingEnv(
    windows_tensor=windows_tensor,
    encoder=encoder
)

# =========================
# LOAD MODEL
# =========================
model = PPO.load("ppo_phase5_continuous_AAPL")

obs, _ = env.reset()

positions = []
prices = []
balances = []

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)

    positions.append(info["position"])
    balances.append(info["balance"])
    prices.append(env._get_price(env.current_step))

# =========================
# PRINT SUMMARY
# =========================
print("FINAL BALANCE:", balances[-1])
print("AVG POSITION :", np.mean(positions))
print("MAX POSITION :", np.max(positions))
