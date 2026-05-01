from stable_baselines3 import PPO
from celeste_ai_gym.CelesteEnv import CelesteEnv
from rllte.xplore.reward import RND
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import torch as th
import gymnasium as gym
import numpy as np
import torch

class DRND(RND):
    def __init__(self, envs, device="cpu"):
        super().__init__(envs, device=device)

        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 1e-4
    
    def update_running_stats(self, x):
        batch_mean = x.mean().item()
        batch_var = x.var().item()
        batch_count = x.numel()

        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        new_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.running_mean = new_mean
        self.running_var = new_var
        self.count = total_count

    def compute(self, samples, sync=True):
        obs = samples["next_observations"].to(self.device).float()

        with torch.no_grad():
            target_features = self.target(obs)

        pred_features = self.predictor(obs)

        intrinsic_reward = (pred_features - target_features).pow(2).mean(dim=1)

        intrinsic_reward = torch.nan_to_num(intrinsic_reward, nan=0.0, posinf=1.0, neginf=-1.0)

        self.update_running_stats(intrinsic_reward)

        intrinsic_reward = (intrinsic_reward - self.running_mean) / (
            (self.running_var ** 0.5) + 1e-8
        )

        intrinsic_reward = (intrinsic_reward - intrinsic_reward.mean()) / (intrinsic_reward.std() + 1e-8)

        intrinsic_reward = torch.clamp(intrinsic_reward, -1.0, 1.0)

        return intrinsic_reward
    


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
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations)
        obs = obs.reshape(-1, *obs.shape[2:])
        # get the new observations
        new_obs = th.as_tensor(self.locals["new_obs"])
        new_obs = new_obs.reshape(-1, *new_obs.shape[1:])
        actions = th.as_tensor(self.buffer.actions).reshape(-1, *self.buffer.actions.shape[2:])
        rewards = th.as_tensor(self.buffer.rewards).reshape(-1)
        dones = th.as_tensor(self.buffer.episode_starts).reshape(-1)
        print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        obs=obs.float()
        # compute the intrinsic rewards
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions, 
                         rewards=rewards, terminateds=dones, 
                         truncateds=dones, next_observations=new_obs),
            sync=True)
        # add the intrinsic rewards to the buffer
        intrinsic_rewards = intrinsic_rewards.detach().cpu().numpy().reshape(-1, 1)
        intrinsic_rewards *= 0.01
        self.buffer.rewards += intrinsic_rewards
        # ===================== compute the intrinsic rewards ===================== #

# Parallel environments
device = "cuda"
n_envs = 1
env = CelesteEnv(render_mode="none")
envs = DummyVecEnv([lambda: env])
envs = VecTransposeImage(envs)

# ===================== build the reward ===================== #
irs = DRND(envs, device="cpu")
# ===================== build the reward ===================== #

model = PPO("CnnPolicy", envs, verbose=1, device="cpu", tensorboard_log ="./drnd_forsaken_tensorboard/", clip_range=0.2)
model.learn(total_timesteps=500000, callback=RLeXploreWithOnPolicyRL(irs))