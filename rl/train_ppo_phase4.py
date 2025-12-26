import os
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.multi_asset_env import MultiAssetTradingEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

# ===============================
# Reproducibility
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===============================
# Load TRAIN assets
# ===============================
DATA_DIR = "data/train"
asset_files = sorted(os.listdir(DATA_DIR))

train_assets = {}
for file in asset_files:
    name = file.replace("_train.pt", "")
    train_assets[name] = torch.load(os.path.join(DATA_DIR, file))

print("Loaded train assets:", list(train_assets.keys()))

# ===============================
# Build shared encoder
# ===============================
sample_tensor = next(iter(train_assets.values()))
num_features = sample_tensor.shape[1]

cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

encoder.eval()  # encoder is NOT trained here

# ===============================
# Environment factory
# ===============================
def make_env():
    return MultiAssetTradingEnv(
        asset_windows_dict=train_assets,
        encoder=encoder
        # lambda_dd is ignored now (dynamic λ is inside TradingEnv)
    )

env = DummyVecEnv([make_env])

# ===============================
# PPO configuration
# ===============================
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    seed=SEED
)

# ===============================
# TRAINING
# ===============================
TOTAL_TIMESTEPS = 300_000  # increase to 600k if you want

print("\n===============================")
print("PHASE 4 TRAINING STARTED")
print("Regime-aware dynamic λ ENABLED")
print("===============================\n")

model.learn(total_timesteps=TOTAL_TIMESTEPS)

# ===============================
# SAVE MODEL
# ===============================
SAVE_PATH = "ppo_phase4_dynamic_lambda"
model.save(SAVE_PATH)

print(f"\nModel saved as: {SAVE_PATH}")
print("PHASE 4 TRAINING COMPLETE")
