import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, sequence_length, obs_shape, action_dim, sprite_slots=0):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.sprite_slots = int(sprite_slots)
        
        # Buffer storage
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminals = np.zeros((capacity,), dtype=np.bool_)
        self.mario_xy = np.zeros((capacity, 2), dtype=np.float32)
        if self.sprite_slots > 0:
            self.sprite_alive = np.zeros((capacity, self.sprite_slots), dtype=np.float32)
            self.sprite_xy = np.zeros((capacity, self.sprite_slots, 2), dtype=np.float32)
        
        self.idx = 0
        self.full = False

    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def add(self, obs, action, reward, terminal, sprite_alive=None, sprite_xy=None, mario_xy=None):
        self.observations[self.idx] = self._to_numpy(obs)
        self.actions[self.idx] = self._to_numpy(action)
        self.rewards[self.idx] = float(reward)
        self.terminals[self.idx] = bool(terminal)
        if mario_xy is not None:
            self.mario_xy[self.idx] = self._to_numpy(mario_xy)
        if self.sprite_slots > 0 and sprite_alive is not None and sprite_xy is not None:
            self.sprite_alive[self.idx] = self._to_numpy(sprite_alive)
            self.sprite_xy[self.idx] = self._to_numpy(sprite_xy)
        
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
        alive_seq = []
        xy_seq = []
        mario_seq = []
        
        for idx in valid_indices:
            obs_seq.append(self.observations[idx : idx + self.sequence_length])
            act_seq.append(self.actions[idx : idx + self.sequence_length])
            rew_seq.append(self.rewards[idx : idx + self.sequence_length])
            term_seq.append(self.terminals[idx : idx + self.sequence_length])
            mario_seq.append(self.mario_xy[idx : idx + self.sequence_length])
            if self.sprite_slots > 0:
                alive_seq.append(self.sprite_alive[idx : idx + self.sequence_length])
                xy_seq.append(self.sprite_xy[idx : idx + self.sequence_length])
            
        batch = (
            torch.from_numpy(np.stack(obs_seq)).float() / 255.0,
            torch.from_numpy(np.stack(act_seq)),
            torch.from_numpy(np.stack(rew_seq)),
            torch.from_numpy(np.stack(term_seq)).float(),
            torch.from_numpy(np.stack(mario_seq)).float(),
        )
        if self.sprite_slots > 0:
            batch = batch + (
                torch.from_numpy(np.stack(alive_seq)).float(),
                torch.from_numpy(np.stack(xy_seq)).float(),
            )
        return batch
