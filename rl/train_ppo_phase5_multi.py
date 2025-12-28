import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.multi_asset_continuous_env import MultiAssetContinuousEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

# ======================
# LOAD DATA
# ======================
DATA_DIR = "data/train"

asset_windows = {
    f.replace("_train.pt", ""): torch.load(os.path.join(DATA_DIR, f))
    for f in os.listdir(DATA_DIR)
    if f.endswith("_train.pt")
}

print("Loaded assets:", list(asset_windows.keys()))

num_features = next(iter(asset_windows.values())).shape[1]

# ======================
# ENCODER
# ======================
cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

for p in encoder.parameters():
    p.requires_grad = False

# ======================
# ENV
# ======================
def make_env():
    return MultiAssetContinuousEnv(
        asset_windows_dict=asset_windows,
        encoder=encoder,
        transaction_cost=0.0005,
        lambda_dd=0.001
    )

env = DummyVecEnv([make_env])

# ======================
# PPO
# ======================
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.99,
    verbose=1,
)

# ======================
# TRAIN
# ======================
model.learn(total_timesteps=600_000)
model.save("ppo_phase5_continuous_multi")

print("PHASE 5 MULTI-ASSET TRAINING COMPLETE")
