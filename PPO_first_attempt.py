from stable_baselines3 import PPO
from celeste_ai_gym.CelesteEnv import CelesteEnv

env = CelesteEnv()
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_celeste_tensorboard/")
model.learn(total_timesteps=100000)