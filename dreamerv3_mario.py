import sys
import os
import torch
import numpy as np
import cv2
import time

# Add local folders to path
sys.path.append(os.path.join(os.getcwd(), 'dreamerv3_scratch'))
sys.path.append(os.path.join(os.getcwd(), 'zelda_gym'))

from mario_bizhawk_env import MarioBizHawkEnv
from agent import DreamerAgent
from buffer import ReplayBuffer

def main():
    # Config
    obs_shape = (1, 64, 64) # Grayscale, 64x64
    action_type = "discrete"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Env
    env = MarioBizHawkEnv(
        rank=0,
        headless=False,
        obs_size=64, # Dreamer likes 64x64
        action_type=action_type,
        frameskip=4
    )
    
    action_dim = env.action_space.n
    
    # Initialize Agent
    agent = DreamerAgent(obs_shape, action_dim, device=device)
    buffer = ReplayBuffer(capacity=100000, sequence_length=32, obs_shape=obs_shape, action_dim=action_dim)
    
    # Training Loop
    total_steps = 1000000
    step = 0
    
    while step < total_steps:
        obs, info = env.reset()
        # obs is (64, 64, 1), convert to (1, 1, 64, 64) for Dreamer
        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0 - 0.5
        
        # Initial RSSM state
        state = agent.rssm.initial_state(1, device)
        action_onehot = torch.zeros(1, action_dim).to(device)
        
        done = False
        episode_reward = 0
        
        while not done:
            # Act
            action, action_onehot, state = agent.act(obs_t, state, action_onehot, training=True)
            
            # Step Env
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Add to buffer
            # buffer expects (A,) for actions
            buffer.add(obs.transpose(2,0,1), action_onehot.cpu().numpy()[0], reward, done)
            
            obs = next_obs
            obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0 - 0.5
            step += 1
            
            # Train
            if step > 1000 and step % 5 == 0:
                batch = buffer.sample(16)
                metrics = agent.train_step(*[t.to(device) for t in batch])
                
                if step % 100 == 0:
                    print(f"Step {step} | Reward: {episode_reward:.2f} | Model Loss: {metrics['loss_model']:.4f} | Actor Loss: {metrics['loss_actor']:.4f}")
                
                if step % 5000 == 0:
                    os.makedirs('checkpoints', exist_ok=True)
                    torch.save(agent.state_dict(), f'checkpoints/dreamer_mario_{step}.pt')
                    print(f"Saved checkpoint at step {step}")


        print(f"Episode Finished | Total Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    main()
