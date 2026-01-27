import torch
import numpy as np
import gymnasium as gym
import cv2
from agent import DreamerAgent
from buffer import ReplayBuffer

class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, *size), dtype=np.uint8
        )

    def observation(self, obs):
        if isinstance(obs, tuple): obs = obs[0]
        # Resize and transpose to CHW
        obs = cv2.resize(obs, self.size, interpolation=cv2.INTER_AREA)
        return obs.transpose(2, 0, 1)

def main():
    # Hyperparameters
    env_name = "CartPole-v1" # Or something with images like "BreakoutNoFrameskip-v4"
    # Note: Dreamer is designed for visual inputs, but can work with vectors.
    # For CartPole, we'd need a different Encoder. 
    # Let's assume an image-based environment for this demonstration.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup environment (Mocking a visual environment)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = ObsWrapper(env)
    
    obs_shape = (3, 64, 64)
    action_dim = env.action_space.n
    
    agent = DreamerAgent(obs_shape, action_dim, device=device)
    buffer = ReplayBuffer(capacity=100000, sequence_length=32, obs_shape=obs_shape, action_dim=action_dim)
    
    # Initial collection
    obs, info = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # One-hot action for buffer
        action_onehot = np.zeros(action_dim)
        action_onehot[action] = 1.0
        
        buffer.add(obs, action_onehot, reward, terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()

    # Training loop
    for step in range(10000):
        # 1. Collect
        action_dist = agent.actor(torch.zeros(1, 512 + 32*32).to(device)) # Dummy state for now
        # In practice, you'd track the state across steps
        # This is simplified
        action = env.action_space.sample() 
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        action_onehot = np.zeros(action_dim)
        action_onehot[action] = 1.0
        buffer.add(obs, action_onehot, reward, terminated or truncated)
        obs = next_obs
        
        if terminated or truncated:
            obs, info = env.reset()
            
        # 2. Train
        if step % 10 == 0:
            batch = buffer.sample(16)
            metrics = agent.train_step(*[t.to(device) for t in batch])
            
            if step % 100 == 0:
                print(f"Step {step}: Model Loss: {metrics['loss_model']:.4f}, Actor Loss: {metrics['loss_actor']:.4f}")

if __name__ == "__main__":
    main()
