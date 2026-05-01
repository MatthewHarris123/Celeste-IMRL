from stable_baselines3 import PPO
from celeste_ai_gym.CelesteEnv import CelesteEnv

env = CelesteEnv(render_mode="none")
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_forsaken_tensorboard/")
model.learn(total_timesteps=500000)