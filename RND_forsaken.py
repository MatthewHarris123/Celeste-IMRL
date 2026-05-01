from stable_baselines3 import PPO
from celeste_ai_gym.CelesteEnv import CelesteEnv
from rllte.xplore.reward import RND
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import torch as th
import gymnasium as gym
import numpy as np

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
        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        obs=obs.float() / 255.0
        # compute the intrinsic rewards
        intrinsic_rewards = irs.compute(
            samples=dict(observations=obs, actions=actions, 
                         rewards=rewards, terminateds=dones, 
                         truncateds=dones, next_observations=new_obs),
            sync=True)
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #

# Parallel environments
device = "cuda"
n_envs = 1
env = CelesteEnv(render_mode="none")
envs = DummyVecEnv([lambda: FloatObsWrapper(env)])
envs = VecTransposeImage(envs)

# ===================== build the reward ===================== #
irs = RND(envs, device="cpu")
# ===================== build the reward ===================== #

model = PPO("CnnPolicy", envs, verbose=1, device="cpu", tensorboard_log ="./rnd_forsaken/tensorboard/")
model.learn(total_timesteps=500000, callback=RLeXploreWithOnPolicyRL(irs))