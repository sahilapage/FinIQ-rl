import os
import torch
import random
import numpy as np

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

print("Loaded assets:", list(train_assets.keys()))

# ===============================
# Build shared encoder
# ===============================
sample_tensor = next(iter(train_assets.values()))
num_features = sample_tensor.shape[1]

cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

# ===============================
# Lambda sweep values
# ===============================
LAMBDA_VALUES = [0.0, 0.0003, 0.001, 0.003, 0.01]

# ===============================
# Training loop
# ===============================
for lambda_dd in LAMBDA_VALUES:

    print("\n" + "=" * 60)
    print(f"TRAINING PPO WITH lambda_dd = {lambda_dd}")
    print("=" * 60)

    def make_env():
        return MultiAssetTradingEnv(
            asset_windows_dict=train_assets,
            encoder=encoder,
            lambda_dd=lambda_dd
        )

    env = DummyVecEnv([make_env])

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

    model.learn(total_timesteps=200_000)

    save_path = f"ppo_lambda_{lambda_dd}"
    model.save(save_path)

    print(f"Model saved as {save_path}")

print("\nALL λ-SWEEP TRAINING COMPLETE")
