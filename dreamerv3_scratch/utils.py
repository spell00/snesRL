import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

class TwoHotDist:
    def __init__(self, logits, dims=0, low=-20, high=20, num_bins=255):
        self.logits = logits
        self.dims = dims
        self.low = low
        self.high = high
        self.num_bins = num_bins
        self.bins = torch.linspace(low, high, num_bins, device=logits.device)

    def mean(self):
        probs = F.softmax(self.logits, dim=-1)
        return torch.sum(probs * self.bins, dim=-1)

    def mode(self):
        # For mode, we can just take the value of the bin with max probability
        # or use the mean. DreamerV3 often uses the mean of the distribution.
        return self.mean()

    def log_prob(self, value):
        # Map value to two-hot representation
        value = torch.clamp(value, self.low, self.high)
        target = (value - self.low) / (self.high - self.low) * (self.num_bins - 1)
        left = torch.floor(target).long()
        right = torch.ceil(target).long()
        
        weight_right = target - left
        weight_left = 1.0 - weight_right
        
        # We want to compute log(p_left * weight_left + p_right * weight_right)
        # But usually we just use cross-entropy against the two-hot target.
        log_probs = F.log_softmax(self.logits, dim=-1)
        
        # Scatter target indices
        # This is equivalent to cross entropy with soft labels
        # target_dist = zeros like logits
        # target_dist.scatter(left, weight_left)
        # target_dist.scatter(right, weight_right)
        # return -(target_dist * log_probs).sum(-1)
        
        # Simplified cross entropy implementation:
        lp_left = torch.gather(log_probs, -1, left.unsqueeze(-1)).squeeze(-1)
        lp_right = torch.gather(log_probs, -1, right.unsqueeze(-1)).squeeze(-1)
        
        return weight_left * lp_left + weight_right * lp_right

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 512], act=nn.SiLU):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(act())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
