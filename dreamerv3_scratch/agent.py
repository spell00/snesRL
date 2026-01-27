import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rssm import RSSM
from networks import Encoder, Decoder, RewardPredictor, ContinuePredictor, Actor, Critic
from utils import symlog, symexp, weight_init

class DreamerAgent(nn.Module):
    def __init__(self, obs_shape, action_dim, lr=1e-4, device='cuda'):
        super().__init__()
        self.device = device
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.lr = lr
        
        # World Model
        self.encoder = Encoder(obs_shape).to(device)
        self.rssm = RSSM(action_dim).to(device)
        self.decoder = Decoder(512 + 32*32, obs_shape).to(device) # deter + stoch
        self.reward_predictor = RewardPredictor(512 + 32*32).to(device)
        self.continue_predictor = ContinuePredictor(512 + 32*32).to(device)
        
        # Actor-Critic
        self.actor = Actor(512 + 32*32, action_dim).to(device)
        self.critic = Critic(512 + 32*32).to(device)
        
        self.apply(weight_init)
        
        # Optimizers
        self.model_opt = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.rssm.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.reward_predictor.parameters()) + 
            list(self.continue_predictor.parameters()), 
            lr=lr, eps=1e-8
        )
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-8)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-8)

    def act(self, obs, prev_state, prev_action, training=True):
        # obs: (1, C, H, W)
        # prev_state: {'stoch', 'deter'}
        # prev_action: (1, A)
        
        with torch.no_grad():
            embed = self.encoder(obs)
            state = self.rssm.observe(embed, prev_action, prev_state)
            feat = torch.cat([state['deter'], state['stoch'].reshape(1, -1)], dim=-1)
            action_dist = self.actor(feat)
            
            if training:
                action = action_dist.sample()
            else:
                if self.actor.continuous:
                    action = action_dist.mean
                else:
                    action = torch.argmax(action_dist.logits, dim=-1)
            
            # One-hot the action for the next step's rssm input
            if not self.actor.continuous:
                action_onehot = F.one_hot(action, self.action_dim).float()
            else:
                action_onehot = action
                
            return action.cpu().numpy()[0], action_onehot, state


    def train_step(self, obs, actions, rewards, terminals):
        # obs: (B, T, C, H, W)
        # actions: (B, T, A)
        # rewards: (B, T)
        # terminals: (B, T)
        
        B, T = obs.shape[:2]
        
        # 1. World Model Training
        embeds = self.encoder(obs.reshape(B*T, *self.obs_shape)).reshape(B, T, -1)
        
        states = []
        state = self.rssm.initial_state(B, self.device)
        
        for t in range(T):
            state = self.rssm.observe(embeds[:, t], actions[:, t], state)
            states.append(state)
            
        # Stack all states
        deter = torch.stack([s['deter'] for s in states], dim=1) # (B, T, D)
        stoch = torch.stack([s['stoch'] for s in states], dim=1) # (B, T, S, C)
        logits = torch.stack([s['logits'] for s in states], dim=1)
        prior_logits = torch.stack([s['prior_logits'] for s in states], dim=1)
        
        # Concatenated state for predictors
        feat = torch.cat([deter, stoch.reshape(B, T, -1)], dim=-1)
        
        # Losses
        # Reconstruction
        recon = self.decoder(feat.reshape(B*T, -1)).reshape(B, T, *self.obs_shape)
        loss_recon = F.mse_loss(recon, obs)
        
        # Reward
        rew_dist = self.reward_predictor(feat.reshape(B*T, -1))
        loss_rew = -rew_dist.log_prob(symlog(rewards.reshape(B*T))).mean()
        
        # Continue
        cont_pred = self.continue_predictor(feat.reshape(B*T, -1))
        loss_cont = F.binary_cross_entropy(cont_pred.reshape(B, T), 1.0 - terminals)
        
        # KL Loss (KL Balancing)
        # KL(post || prior)
        kl_loss = self.compute_kl_loss(logits, prior_logits)
        
        loss_model = loss_recon + loss_rew + loss_cont + 0.5 * kl_loss
        
        self.model_opt.zero_grad()
        loss_model.backward()
        self.model_opt.step()
        
        # 2. Actor-Critic Training (Imagination)
        # Start from the posterior states discovered during WM training
        # Usually we detach the features to not backprop into WM
        start_feat = feat.detach().reshape(B*T, -1)
        # Run imagination for horizon H
        hor = 15
        img_feats = []
        curr_state = {
            'deter': deter.detach().reshape(B*T, -1),
            'stoch': stoch.detach().reshape(B*T, 32, 32)
        }
        
        for _ in range(hor):
            feat_t = torch.cat([curr_state['deter'], curr_state['stoch'].reshape(B*T, -1)], dim=-1)
            img_feats.append(feat_t)
            
            action_dist = self.actor(feat_t.detach())
            action = action_dist.sample()
            # If discrete, handle one-hot
            if not self.actor.continuous:
                action = F.one_hot(action, self.action_dim).float()
                
            curr_state = self.rssm.imagine(action, curr_state)
            
        img_feats = torch.stack(img_feats, dim=1) # (B*T, hor, F)
        
        # Predict rewards and values for imagined states
        img_rew_dist = self.reward_predictor(img_feats)
        img_rews = symexp(img_rew_dist.mean())
        
        img_val_dist = self.critic(img_feats)
        img_vals = symexp(img_val_dist.mean())
        
        # Compute Lambda-Values
        returns = self.compute_lambda_values(img_rews, img_vals)
        
        # Actor Loss
        # Maximize returns
        loss_actor = -returns.mean()
        
        # Critic Loss
        # Match lambda-values
        val_dist = self.critic(img_feats.detach())
        loss_critic = -val_dist.log_prob(symlog(returns.detach())).mean()
        
        self.actor_opt.zero_grad()
        loss_actor.backward()
        self.actor_opt.step()
        
        self.critic_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()
        
        return {
            'loss_model': loss_model.item(),
            'loss_recon': loss_recon.item(),
            'loss_rew': loss_rew.item(),
            'loss_kl': kl_loss.item(),
            'loss_actor': loss_actor.item(),
            'loss_critic': loss_critic.item()
        }

    def compute_kl_loss(self, post_logits, prior_logits):
        # KL Balancing: 0.1 for posterior, 0.9 for prior (usually)
        # DreamerV3 uses slightly different balancing
        # KL(post || prior)
        post_dist = torch.distributions.Independent(torch.distributions.Categorical(logits=post_logits), 1)
        prior_dist = torch.distributions.Independent(torch.distributions.Categorical(logits=prior_logits), 1)
        
        # Detached versions for KL balancing
        post_dist_detached = torch.distributions.Independent(torch.distributions.Categorical(logits=post_logits.detach()), 1)
        prior_dist_detached = torch.distributions.Independent(torch.distributions.Categorical(logits=prior_logits.detach()), 1)
        
        kl_post = torch.distributions.kl.kl_divergence(post_dist, prior_dist_detached)
        kl_prior = torch.distributions.kl.kl_divergence(post_dist_detached, prior_dist)
        
        # 0.1 * kl_post + 0.9 * kl_prior
        return 0.1 * kl_post.mean() + 0.9 * kl_prior.mean()

    def compute_lambda_values(self, rews, vals, gamma=0.99, lmbda=0.95):
        # rews: (B, H), vals: (B, H)
        # returns: (B, H)
        # last value is bootstrapping
        H = rews.shape[1]
        returns = torch.zeros_like(rews)
        last_val = vals[:, -1]
        
        for t in reversed(range(H - 1)):
            last_val = rews[:, t] + gamma * ((1 - lmbda) * vals[:, t+1] + lmbda * last_val)
            returns[:, t] = last_val
        
        return returns
