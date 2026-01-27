# dreamer_mario.py
# DreamerV3 training script for MarioBizHawkEnv
# Requires: https://github.com/danijar/dreamerv3 and JAX

import embodied
import dreamerv3
from mario_bizhawk_env import MarioBizHawkEnv

class MarioDreamerEnv(embodied.Env):
    def __init__(self, **kwargs):
        self.env = MarioBizHawkEnv(**kwargs)
        self.obs_space = {
            'image': embodied.Space(np.uint8, self.env.observation_space.shape)
        }
        self.act_space = {
            'action': embodied.Space(np.int32, self.env.action_space.shape)
        }

    def step(self, action):
        act = action['action'] if isinstance(action, dict) else action
        obs, reward, terminated, truncated, info = self.env.step(act)
        done = terminated or truncated
        return {
            'image': obs,
            'reward': float(reward),
            'is_first': False,
            'is_last': done,
            'is_terminal': terminated,
        }

    def reset(self):
        obs, info = self.env.reset()
        return {
            'image': obs,
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }

if __name__ == "__main__":
    import numpy as np
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='./logs/dreamer_mario')
    parser.add_argument('--timesteps', type=int, default=1000000)
    # Add more arguments as needed for MarioBizHawkEnv
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    config = embodied.Config(
        'dreamerv3',
        run.logdir=args.logdir,
        run.train_ratio=64,
        run.steps=args.timesteps,
        # Add more DreamerV3 config options as needed
    )

    env = MarioDreamerEnv()
    agent = dreamerv3.Agent(config, env.obs_space, env.act_space)
    embodied.run.train(agent, env, config)
