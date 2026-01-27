import sys
import os
import torch
import numpy as np
import time
import functools

# Add local folders to path
sys.path.append(os.path.join(os.getcwd(), 'dreamerv3_scratch'))
sys.path.append(os.path.join(os.getcwd(), 'zelda_gym'))

from mario_bizhawk_env import MarioBizHawkEnv
from agent import DreamerAgent
from buffer import ReplayBuffer
from parallel_envs import ParallelEnvs

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel training environments")
    parser.add_argument("--total_steps", type=int, default=2000000, help="Total training steps")
    parser.add_argument("--frameskip", type=int, default=4, help="Frameskip for the environment")
    parser.add_argument("--obs_size", type=int, default=64, help="Observation size")
    parser.add_argument("--death_penalty", type=float, default=-50.0, help="Penalty for dying")
    parser.add_argument("--exploration_bonus", type=float, default=0.5, help="Exploration bonus amount")
    parser.add_argument("--enable_cell_exploration", action="store_true", default=True, help="Enable cell-based exploration")
    parser.add_argument("--level", type=int, default=3, help="Level index: 0=random, 1-N=specific savestate")
    parser.add_argument("--no_load", action="store_true", help="Do not load the latest checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def make_env(rank, args):
    env = MarioBizHawkEnv(
        rank=rank,
        headless=False,
        obs_size=args.obs_size,
        action_type="discrete",
        frameskip=args.frameskip,
        death_penalty=args.death_penalty,
        exploration_bonus=args.exploration_bonus,
        enable_cell_exploration=args.enable_cell_exploration,
        startup_sleep_s=10.0
    )
    
    # Wrap reset to handle level selection
    if args.level != 0:
        orig_reset = env.reset
        level_index = args.level # capture current level

        def reset_with_level(*, seed=None, options=None):
            opts = dict(options) if options else {}
            opts["savestate"] = env.get_savestate_by_index(level_index)
            return orig_reset(seed=seed, options=opts)
        
        env.reset = reset_with_level
    
    return env

def latest_checkpoint():
    if not os.path.exists('checkpoints'):
        return None
    ckpts = [f for f in os.listdir('checkpoints') if f.startswith("dreamer_mario_multi_") and f.endswith(".pt")]
    if not ckpts:
        return None
    try:
        # Sort by step number: dreamer_mario_multi_5000.pt -> 5000
        ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join('checkpoints', ckpts[-1])
    except:
        return None

def main():
    args = parse_args()
    n_envs = args.n_envs
    obs_shape = (1, args.obs_size, args.obs_size) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} with {n_envs} environments")

    # Initialize Parallel Envs
    dummy_env = make_env(999, args) # Temp to get action_dim
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    env_fn = functools.partial(make_env, args=args)
    envs = ParallelEnvs(n_envs, env_fn, obs_shape, action_dim, device)
    
    # Initialize Agent
    agent = DreamerAgent(obs_shape, action_dim, lr=1e-4, device=device)
    
    # Load latest checkpoint if exists
    if not args.no_load:
        ckpt = latest_checkpoint()
        if ckpt:
            print(f"Loading checkpoint: {ckpt}")
            agent.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print("No checkpoint found. Starting from scratch.")
    else:
        print("Starting FRESH training (--no_load).")

    # Increased sequence length for better memory
    buffer = ReplayBuffer(capacity=100000, sequence_length=64, obs_shape=obs_shape, action_dim=action_dim)
    
    # Training Loop
    total_steps = args.total_steps
    step = 0
    
    obs, infos = envs.reset()
    # obs shape: (N, 64, 64, 1) -> (N, 1, 64, 64)
    obs_t = torch.from_numpy(obs).float().permute(0, 3, 1, 2).to(device) / 255.0 - 0.5
    
    # States for each env
    states = agent.rssm.initial_state(n_envs, device)
    action_onehots = torch.zeros(n_envs, action_dim).to(device)
    
    print("Starting collection...")
    rewards = []
    while step < total_steps:
        # Act
        # agent.act usually expects (1, C, H, W). We need to batch it.
        # Let's use the underlying modules for batch inference
        with torch.no_grad():
            embeds = agent.encoder(obs_t)
            states = agent.rssm.observe(embeds, action_onehots, states)
            feats = torch.cat([states['deter'], states['stoch'].reshape(n_envs, -1)], dim=-1)
            action_dists = agent.actor(feats)
            actions = action_dists.sample()
            action_onehots = torch.nn.functional.one_hot(actions, action_dim).float()
            
        actions_np = actions.cpu().numpy()
        
        # Step Envs
        next_obs, reward, terms, truncs, infos = envs.step(actions_np)
        
        # Add to buffer
        for i in range(n_envs):
            # Transpose to CHW for buffer storage
            buffer.add(obs[i].transpose(2,0,1), action_onehots[i].cpu().numpy(), reward[i], terms[i] or truncs[i])
        
        obs = next_obs
        obs_t = torch.from_numpy(obs).float().permute(0, 3, 1, 2).to(device) / 255.0 - 0.5
        step += n_envs
        
        # Reset RSSM state for finished episodes
        for i in range(n_envs):
            if terms[i] or truncs[i]:
                # This is tricky: we'd ideally reset only the i-th state.
                # For now, let's just clear the hidden state for that index.
                # A more robust implementation would handle per-env state resets properly.
                # Here we just zero it out for simplicity in scratch impl.
                states['deter'][i] = 0
                states['stoch'][i] = 0
                action_onehots[i] = 0

        # Train
        if step > 2000 and step % (n_envs * 5) == 0:
            rewards.append(reward)
            batch = buffer.sample(16)
            metrics = agent.train_step(*[t.to(device) for t in batch])
            
            if step % 100 < n_envs:
                # Calculate mean across the collected list of reward-batches
                avg_reward = np.mean(rewards) if rewards else 0.0
                print(f"Step {step} | Avg Step Reward: {avg_reward:.4f} | Model Loss: {metrics['loss_model']:.4f} | Actor Loss: {metrics['loss_actor']:.4f}")
                rewards = [] # Reset after printing/logging
                
                if step % 5000 < n_envs:
                    os.makedirs('checkpoints', exist_ok=True)
                    torch.save(agent.state_dict(), f'checkpoints/dreamer_mario_multi_{step}.pt')

if __name__ == "__main__":
    main()
