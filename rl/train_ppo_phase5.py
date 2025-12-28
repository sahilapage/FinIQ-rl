import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.trading_env_continuous import ContinuousTradingEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

# =========================
# LOAD DATA
# =========================
windows_tensor = torch.load("data/train/AAPL_train.pt")

num_features = windows_tensor.shape[1]

# =========================
# BUILD ENCODER
# =========================
cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

# Freeze encoder (important)
for p in encoder.parameters():
    p.requires_grad = False

# =========================
# ENV
# =========================
def make_env():
    return ContinuousTradingEnv(
        windows_tensor=windows_tensor,
        encoder=encoder,
        transaction_cost=0.0005,
        lambda_dd=0.001
    )

env = DummyVecEnv([make_env])

# =========================
# PPO MODEL
# =========================
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,      # exploration is critical in continuous control
    gamma=0.99,
    verbose=1,
)

# =========================
# TRAIN
# =========================
model.learn(total_timesteps=300_000)

model.save("ppo_phase5_continuous_AAPL")

print("PHASE 5 (Single Asset) TRAINING COMPLETE")
