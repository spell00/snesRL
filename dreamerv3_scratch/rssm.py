import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent

class RSSM(nn.Module):
    def __init__(self, action_dim, stoch_dim=32, stoch_classes=32, deter_dim=512, hidden_dim=512, rssm_type="categorical"):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.deter_dim = deter_dim
        self.rssm_type = rssm_type
        
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        
        # Prior/Transition predictor
        if self.rssm_type == "categorical":
            self.img_in = nn.Linear(stoch_dim * stoch_classes + action_dim, hidden_dim)
            self.img_out = nn.Linear(deter_dim, stoch_dim * stoch_classes)
            # Posterior/Representation predictor
            self.obs_out = nn.Linear(deter_dim + hidden_dim, stoch_dim * stoch_classes)
        elif self.rssm_type == "gaussian":
            # For Gaussian, stoch is just stoch_dim size (continuous vector)
            # Input to RNN is stoch_dim + action_dim
            self.img_in = nn.Linear(stoch_dim + action_dim, hidden_dim)
            # Output is mean + std (2 * stoch_dim)
            self.img_out = nn.Linear(deter_dim, 2 * stoch_dim)
            # Posterior
            self.obs_out = nn.Linear(deter_dim + hidden_dim, 2 * stoch_dim)
        else:
            raise ValueError(f"Unknown rssm_type: {rssm_type}")

    def initial_state(self, batch_size, device):
        if self.rssm_type == "categorical":
            return {
                'stoch': torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device),
                'deter': torch.zeros(batch_size, self.deter_dim, device=device)
            }
        else:
             return {
                'stoch': torch.zeros(batch_size, self.stoch_dim, device=device),
                'deter': torch.zeros(batch_size, self.deter_dim, device=device)
            }

    def observe(self, embed, action, prev_state):
        # embed: (B, hidden_dim) - output of encoder
        # action: (B, action_dim)
        # prev_state: {'stoch', 'deter'}
        
        # 1. Prediction (Prior)
        prior_state = self.imagine(action, prev_state)
        
        # 2. Update (Posterior)
        # Use embedding to refine the state
        x = torch.cat([prior_state['deter'], embed], dim=-1)
        stats = self.obs_out(x)
        
        if self.rssm_type == "categorical":
            logits = stats.view(-1, self.stoch_dim, self.stoch_classes)
            post_stoch = self.get_stoch(logits)
            return {
                'stoch': post_stoch,
                'deter': prior_state['deter'],
                'logits': logits,
                'prior_logits': prior_state['logits']
            }
        else:
            # Gaussian: stats is [mean, std]
            mean, std = torch.chunk(stats, 2, dim=-1)
            std = F.softplus(std) + 0.1
            dist = torch.distributions.Normal(mean, std)
            post_stoch = dist.rsample()
            return {
                'stoch': post_stoch,
                'deter': prior_state['deter'],
                'mean': mean,
                'std': std,
                'prior_mean': prior_state['mean'],
                'prior_std': prior_state['std']
            }

    def imagine(self, action, prev_state):
        # action: (B, action_dim)
        # prev_state: {'stoch', 'deter'}
        
        if self.rssm_type == "categorical":
            stoch = prev_state['stoch'].view(-1, self.stoch_dim * self.stoch_classes)
        else:
            stoch = prev_state['stoch'] # (B, stoch_dim)

        x = torch.cat([stoch, action], dim=-1)
        x = F.silu(self.img_in(x))
        deter = self.gru(x, prev_state['deter'])
        
        stats = self.img_out(deter)
        
        if self.rssm_type == "categorical":
            logits = stats.view(-1, self.stoch_dim, self.stoch_classes)
            stoch = self.get_stoch(logits)
            return {
                'stoch': stoch,
                'deter': deter,
                'logits': logits
            }
        else:
            mean, std = torch.chunk(stats, 2, dim=-1)
            std = F.softplus(std) + 0.1
            dist = torch.distributions.Normal(mean, std)
            stoch = dist.rsample()
            return {
                'stoch': stoch,
                'deter': deter,
                'mean': mean,
                'std': std
            }

    def get_stoch(self, logits):
        # Straight-through estimator for discrete latents
        dist = Independent(Categorical(logits=logits), 1)
        stoch = dist.sample()
        # One-hot representation
        stoch_onehot = F.one_hot(stoch, self.stoch_classes).float()
        # Add gradient pass-through
        probs = F.softmax(logits, dim=-1)
        stoch_onehot = stoch_onehot + (probs - probs.detach())
        return stoch_onehot
