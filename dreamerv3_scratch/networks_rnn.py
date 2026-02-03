import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent


class RSSMBase(nn.Module):
    def __init__(self, action_dim, stoch_dim=32, stoch_classes=32, deter_dim=512, hidden_dim=512):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.img_in = nn.Linear(stoch_dim * stoch_classes + action_dim, hidden_dim)
        self.img_out = nn.Linear(deter_dim, stoch_dim * stoch_classes)
        self.obs_out = nn.Linear(deter_dim + hidden_dim, stoch_dim * stoch_classes)

    def initial_state(self, batch_size, device):
        raise NotImplementedError

    def observe(self, embed, action, prev_state):
        prior_state = self.imagine(action, prev_state)
        x = torch.cat([prior_state['deter'], embed], dim=-1)
        logits = self.obs_out(x)
        logits = logits.view(-1, self.stoch_dim, self.stoch_classes)
        post_stoch = self.get_stoch(logits)
        state = {
            'stoch': post_stoch,
            'deter': prior_state['deter'],
            'logits': logits,
            'prior_logits': prior_state['logits']
        }
        if 'cell' in prior_state:
            state['cell'] = prior_state['cell']
        return state

    def imagine(self, action, prev_state):
        raise NotImplementedError

    def get_stoch(self, logits):
        dist = Independent(Categorical(logits=logits), 1)
        stoch = dist.sample()
        stoch_onehot = F.one_hot(stoch, self.stoch_classes).float()
        probs = F.softmax(logits, dim=-1)
        stoch_onehot = stoch_onehot + (probs - probs.detach())
        return stoch_onehot


class RSSMGRU(RSSMBase):
    def __init__(self, action_dim, stoch_dim=32, stoch_classes=32, deter_dim=512, hidden_dim=512):
        super().__init__(action_dim, stoch_dim, stoch_classes, deter_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

    def initial_state(self, batch_size, device):
        return {
            'stoch': torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device),
            'deter': torch.zeros(batch_size, self.deter_dim, device=device)
        }

    def imagine(self, action, prev_state):
        stoch = prev_state['stoch'].view(-1, self.stoch_dim * self.stoch_classes)
        x = torch.cat([stoch, action], dim=-1)
        x = F.silu(self.img_in(x))
        deter = self.gru(x, prev_state['deter'])
        logits = self.img_out(deter)
        logits = logits.view(-1, self.stoch_dim, self.stoch_classes)
        stoch = self.get_stoch(logits)
        return {
            'stoch': stoch,
            'deter': deter,
            'logits': logits
        }


class RSSMLSTM(RSSMBase):
    def __init__(self, action_dim, stoch_dim=32, stoch_classes=32, deter_dim=512, hidden_dim=512):
        super().__init__(action_dim, stoch_dim, stoch_classes, deter_dim, hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim, deter_dim)

    def initial_state(self, batch_size, device):
        return {
            'stoch': torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device),
            'deter': torch.zeros(batch_size, self.deter_dim, device=device),
            'cell': torch.zeros(batch_size, self.deter_dim, device=device)
        }

    def imagine(self, action, prev_state):
        stoch = prev_state['stoch'].view(-1, self.stoch_dim * self.stoch_classes)
        x = torch.cat([stoch, action], dim=-1)
        x = F.silu(self.img_in(x))
        deter, cell = self.lstm(x, (prev_state['deter'], prev_state['cell']))
        logits = self.img_out(deter)
        logits = logits.view(-1, self.stoch_dim, self.stoch_classes)
        stoch = self.get_stoch(logits)
        return {
            'stoch': stoch,
            'deter': deter,
            'cell': cell,
            'logits': logits
        }