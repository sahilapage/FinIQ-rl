import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.multi_asset_vol_target_env import MultiAssetVolTargetEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

# -----------------------------
# Load assets
# -----------------------------
asset_names = ["AAPL", "MSFT", "NVDA", "TSLA"]
asset_tensors = {}

for name in asset_names:
    asset_tensors[name] = torch.load(f"data/train/{name}_train.pt")

num_features = asset_tensors[asset_names[0]].shape[1]

cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

def make_env():
    return MultiAssetVolTargetEnv(
        asset_tensors=asset_tensors,
        encoder=encoder,
    )

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    ent_coef=0.01,      # IMPORTANT
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
)

model.learn(total_timesteps=600_000)
model.save("ppo_phase5_5_multi")

print("PHASE 5.5 MULTI-ASSET TRAINING COMPLETE")
