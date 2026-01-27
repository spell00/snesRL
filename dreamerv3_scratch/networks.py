import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MLP, TwoHotDist, symlog, symexp

class Encoder(nn.Module):
    def __init__(self, obs_shape, embed_dim=512):
        super().__init__()
        # Simple CNN encoder for 64x64 or similar images
        self.conv1 = nn.Conv2d(obs_shape[0], 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        
        # Flatten and project to embed_dim
        # For 64x64, output of conv4 is usually 2x2x256 = 1024
        self.fc = nn.Linear(1024, embed_dim)

    def forward(self, x):
        # Normalize image to [-0.5, 0.5] if they are [0, 255]
        # x = x.float() / 255.0 - 0.5 
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, state_dim, obs_shape):
        super().__init__()
        self.fc = nn.Linear(state_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, obs_shape[0], 6, stride=2)
        # This is a bit arbitrary, but roughly maps state back to image
        # In practice, DreamerV3 architecture is more specific about kernel sizes

    def forward(self, state):
        x = self.fc(state)
        x = x.view(-1, 1024, 1, 1)
        x = F.silu(self.deconv1(x))
        x = F.silu(self.deconv2(x))
        x = F.silu(self.deconv3(x))
        return self.deconv4(x)

class RewardPredictor(nn.Module):
    def __init__(self, state_dim, num_bins=255):
        super().__init__()
        self.mlp = MLP(state_dim, num_bins)

    def forward(self, state):
        logits = self.mlp(state)
        return TwoHotDist(logits)

class ContinuePredictor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.mlp = MLP(state_dim, 1)

    def forward(self, state):
        return torch.sigmoid(self.mlp(state))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False):
        super().__init__()
        self.continuous = continuous
        if continuous:
            self.mlp = MLP(state_dim, action_dim * 2) # mean and std
        else:
            self.mlp = MLP(state_dim, action_dim) # logits

    def forward(self, state):
        out = self.mlp(state)
        if self.continuous:
            mean, std = torch.chunk(out, 2, dim=-1)
            std = F.softplus(std) + 0.1
            return torch.distributions.Normal(mean, std)
        else:
            return torch.distributions.Categorical(logits=out)

class Critic(nn.Module):
    def __init__(self, state_dim, num_bins=255):
        super().__init__()
        self.mlp = MLP(state_dim, num_bins)

    def forward(self, state):
        logits = self.mlp(state)
        return TwoHotDist(logits)
