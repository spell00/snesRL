import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, sequence_length, obs_shape, action_dim):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # Buffer storage
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminals = np.zeros((capacity,), dtype=np.bool_)
        
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, terminal):
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.terminals[self.idx] = terminal
        
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size):
        # We need to sample sequences of length sequence_length
        # and ensure they don't cross the terminal boundary or buffer boundary
        curr_size = self.capacity if self.full else self.idx
        valid_indices = []
        
        while len(valid_indices) < batch_size:
            start_idx = np.random.randint(0, curr_size - self.sequence_length)
            # Check if sequence crosses terminal (except at the very end)
            if not np.any(self.terminals[start_idx : start_idx + self.sequence_length - 1]):
                valid_indices.append(start_idx)
        
        obs_seq = []
        act_seq = []
        rew_seq = []
        term_seq = []
        
        for idx in valid_indices:
            obs_seq.append(self.observations[idx : idx + self.sequence_length])
            act_seq.append(self.actions[idx : idx + self.sequence_length])
            rew_seq.append(self.rewards[idx : idx + self.sequence_length])
            term_seq.append(self.terminals[idx : idx + self.sequence_length])
            
        return (
            torch.from_numpy(np.stack(obs_seq)).float() / 255.0 - 0.5,
            torch.from_numpy(np.stack(act_seq)),
            torch.from_numpy(np.stack(rew_seq)),
            torch.from_numpy(np.stack(term_seq)).float()
        )
