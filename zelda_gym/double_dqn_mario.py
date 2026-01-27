"""
double_dqn_mario.py
Double DQN training for Mario using BizHawk environment.
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import torch as th

from mario_bizhawk_env import MarioBizHawkEnv


def make_env(rank, **kwargs):
    def _init():
        env = MarioBizHawkEnv(rank=rank, **kwargs)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    n_envs = 4
    env_kwargs = dict(
        frameskip=6,
        screenshot_every=2,
        verbose=2,
        action_type="discrete",
        enable_cell_exploration=True,
        novelty_enabled=True,
        cell_bonus_mode="other",
        use_progress_savestate=True,
    )
    env = SubprocVecEnv([make_env(i, **env_kwargs) for i in range(n_envs)])

    policy_kwargs = dict(
        net_arch=[256, 256],
        dueling=True,
        double_q=True,  # Double DQN
    )

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./dqn_mario_tensorboard/",
        device="auto",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000, save_path="./models/double_dqn/", name_prefix="double_dqn_mario"
    )

    model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
