from stable_baselines3 import PPO
from celeste_ai_gym.CelesteEnv import CelesteEnv
from rllte.xplore.reward import RND
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import torch as th
import gymnasium as gym
import numpy as np
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim=512):
        super().__init__()
        C, H, W = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            dummy = th.zeros(1, C, H, W)
            n_flatten = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(self.conv(x))

class ETD:
    def __init__(self, obs_shape, device):
        self.device = device
        
        self.encoder = CNNEncoder(obs_shape).to(device)
        self.distance_head = th.nn.Sequential(
            th.nn.Linear(512 * 2, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 1)
        ).to(device)

        self.optimizer = th.optim.Adam(
            list(self.encoder.parameters()) + list(self.distance_head.parameters()),
            lr=1e-4
        )

class FloatObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs.astype(np.float32) / 255.0

class RLeXploreWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"]
        device = observations.device
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations)

        # flatten env dimension if needed
        obs = obs.reshape(-1, *obs.shape[2:])
        T = obs.shape[0]

        # sample random pairs
        num_pairs = 1024
        max_gap = 50
        i = th.randint(0, T, (num_pairs,))
        j = i + th.randint(-max_gap, max_gap, (num_pairs,))
        j = j.clamp(0, T - 1)   

        obs_i = obs[i].float()
        obs_j = obs[j].float()

        time_dist = (j - i).abs().float().unsqueeze(1) / T

        z_i = self.irs.encoder(obs_i)
        z_j = self.irs.encoder(obs_j)

        pair = th.cat([z_i, z_j], dim=1)
        pred_dist = self.irs.distance_head(pair)

        pair_error = (pred_dist - time_dist).pow(2).detach()

        intrinsic_rewards = th.zeros(T, device=obs.device)

        intrinsic_rewards.index_add_(0, i, pair_error.squeeze())
        intrinsic_rewards.index_add_(0, j, pair_error.squeeze())

        intrinsic_rewards = intrinsic_rewards / (intrinsic_rewards.mean() + 1e-8)

        dones = th.as_tensor(self.buffer.episode_starts).reshape(-1)
        intrinsic_rewards[dones == 1] = 0.0

        intrinsic_rewards = intrinsic_rewards.unsqueeze(1)
        loss = ((pred_dist - time_dist) ** 2).mean()

        self.irs.optimizer.zero_grad()
        loss.backward()
        self.irs.optimizer.step()

        # encode full trajectory
        z = self.irs.encoder(obs.float())

        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.detach().cpu().numpy()
        self.buffer.returns += intrinsic_rewards.detach().cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #

# Parallel environments
device = "cuda"
n_envs = 1
env = CelesteEnv(render_mode="none")
envs = DummyVecEnv([lambda: FloatObsWrapper(env)])
envs = VecTransposeImage(envs)

# ===================== build the reward ===================== #
obs_shape = envs.observation_space.shape
irs = ETD(obs_shape, device="cpu")
# ===================== build the reward ===================== #

model = PPO("CnnPolicy", envs, verbose=1, device="cpu", tensorboard_log ="./etd_forsaken/tensorboard/")
model.learn(total_timesteps=500000, callback=RLeXploreWithOnPolicyRL(irs))