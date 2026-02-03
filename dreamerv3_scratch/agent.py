import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rssm import RSSM
from networks_rnn import RSSMGRU, RSSMLSTM
from networks import Encoder, Decoder, RewardPredictor, ContinuePredictor, Actor, Critic
from utils import symlog, symexp, weight_init, MLP
from losses import build_box_mask, dice_loss, mask_iou as mask_iou_metric, mask_bce_loss

class DreamerAgent(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        lr=1e-4,
        device='cuda',
        kl_scale=1.0,
        free_nats=1.0,
        recon_weight=1.0,
        reward_weight=1.0,
        continue_weight=1.0,
        delta_recon_weight=1.0,
        edge_recon_weight=1.0,
        fg_recon_weight=1.0,
        fg_recon_threshold=0.1,
        sprite_slots=0,
        sprite_alive_weight=1.0,
        sprite_xy_weight=0.5,
        box_recon_weight=1.0,
        mask_bce_weight=1.0,
        mask_dice_weight=1.0,
        entropy_scale=3e-4,
        recurrent_model="rssm",
        rssm_type="categorical",
        imagination_horizon=90,
    ):
        super().__init__()
        self.device = device
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.lr = lr
        self.kl_scale = kl_scale
        self.free_nats = free_nats
        self.recon_weight = recon_weight
        self.reward_weight = reward_weight
        self.continue_weight = continue_weight
        self.delta_recon_weight = delta_recon_weight
        self.edge_recon_weight = edge_recon_weight
        self.fg_recon_weight = fg_recon_weight
        self.fg_recon_threshold = fg_recon_threshold
        self.sprite_slots = int(sprite_slots)
        self.sprite_alive_weight = sprite_alive_weight
        self.sprite_xy_weight = sprite_xy_weight
        self.box_recon_weight = box_recon_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight
        self.entropy_scale = entropy_scale
        self.imagination_horizon = int(imagination_horizon)
        
        # World Model
        self.encoder = Encoder(obs_shape).to(device)
        if recurrent_model in {"gru", "lstm"} and rssm_type != "categorical":
            raise ValueError(
                f"rssm_type={rssm_type} is only supported for recurrent_model='rssm'."
            )
        if recurrent_model == "gru":
            self.rssm = RSSMGRU(action_dim).to(device)
        elif recurrent_model == "lstm":
            self.rssm = RSSMLSTM(action_dim).to(device)
        else:
            self.rssm = RSSM(action_dim, rssm_type=rssm_type).to(device)
        self.decoder = Decoder(512 + 32*32, obs_shape).to(device) # deter + stoch
        self.reward_predictor = RewardPredictor(512 + 32*32).to(device)
        self.continue_predictor = ContinuePredictor(512 + 32*32).to(device)
        if self.sprite_slots > 0:
            self.sprite_head = MLP(512 + 32*32, self.sprite_slots * 3).to(device)
            self.yolo_slots = self.sprite_slots + 1
            self.yolo_head = MLP(512 + 32*32, self.yolo_slots * 5).to(device)
            self.mask_head = nn.Sequential(
                nn.Linear(512 + 32*32, 1024),
                nn.SiLU(),
                nn.Linear(1024, obs_shape[1] * obs_shape[2])
            ).to(device)
        else:
            self.sprite_head = None
            self.yolo_slots = 0
            self.yolo_head = None
            self.mask_head = None
        
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
            list(self.continue_predictor.parameters()) +
            (list(self.mask_head.parameters()) if self.mask_head is not None else []),
            lr=lr, eps=1e-8
        )
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-8)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-8)
        if self.yolo_head is not None:
            self.yolo_opt = optim.Adam(self.yolo_head.parameters(), lr=lr, eps=1e-8)
        else:
            self.yolo_opt = None
        self.mask_opt = None

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


    def train_step(self, obs, actions, rewards, terminals, mario_xy=None, sprite_alive=None, sprite_xy=None):
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
        cell = None
        if 'cell' in states[-1]:
            cell = torch.stack([s['cell'] for s in states], dim=1)
        logits = torch.stack([s['logits'] for s in states], dim=1)
        prior_logits = torch.stack([s['prior_logits'] for s in states], dim=1)
        
        # Concatenated state for predictors
        feat = torch.cat([deter, stoch.reshape(B, T, -1)], dim=-1)
        
        # Losses
        # Reconstruction
        recon_logits = self.decoder(feat.reshape(B*T, -1)).reshape(B, T, *self.obs_shape)
        recon = recon_logits.sigmoid()
        recon_err = F.mse_loss(recon, obs, reduction='none')
        if (
            self.box_recon_weight > 1.0 and
            sprite_xy is not None and
            mario_xy is not None
        ):
            _, _, _, H, W = recon_err.shape
            mask = torch.ones((B, T, 1, H, W), device=obs.device)
            x_scale = (W - 1) / float(max(256 - 1, 1))
            y_scale = (H - 1) / float(max(224 - 1, 1))
            for b in range(B):
                for t in range(T):
                    if sprite_alive is not None:
                        alive = sprite_alive[b, t]
                        coords = sprite_xy[b, t]
                        for s in range(self.sprite_slots):
                            if alive[s] > 0.5:
                                x0 = int((coords[s, 0]) * x_scale)
                                y0 = int((coords[s, 1]) * y_scale)
                                x1 = int((coords[s, 0] + 16.0) * x_scale)
                                y1 = int((coords[s, 1] + 16.0) * y_scale)
                                x0 = max(min(x0, W - 1), 0)
                                x1 = max(min(x1, W - 1), 0)
                                y0 = max(min(y0, H - 1), 0)
                                y1 = max(min(y1, H - 1), 0)
                                mask[b, t, 0, y0:y1 + 1, x0:x1 + 1] = self.box_recon_weight
                    if mario_xy is not None:
                        mx = mario_xy[b, t, 0]
                        my = mario_xy[b, t, 1]
                        x0 = int((mx) * x_scale)
                        y0 = int((my) * y_scale)
                        x1 = int((mx + 16.0) * x_scale)
                        y1 = int((my + 32.0) * y_scale)
                        x0 = max(min(x0, W - 1), 0)
                        x1 = max(min(x1, W - 1), 0)
                        y0 = max(min(y0, H - 1), 0)
                        y1 = max(min(y1, H - 1), 0)
                        mask[b, t, 0, y0:y1 + 1, x0:x1 + 1] = self.box_recon_weight
            recon_err = recon_err * mask
        loss_recon = recon_err.mean()

        # Delta reconstruction (emphasize sprite motion)
        if T > 1:
            obs_delta = obs[:, 1:] - obs[:, :-1]
            recon_delta = recon[:, 1:] - recon[:, :-1]
            loss_recon_delta = F.mse_loss(recon_delta, obs_delta)
        else:
            loss_recon_delta = torch.tensor(0.0, device=obs.device)

        # Edge/gradient reconstruction (emphasize sharp sprites)
        def grad(x):
            dx = x[..., :, :, 1:] - x[..., :, :, :-1]
            dy = x[..., :, 1:, :] - x[..., :, :-1, :]
            return dx.abs().mean() + dy.abs().mean()
        loss_recon_edge = (grad(obs) - grad(recon)).abs().mean()

        # Foreground mask loss (focus on moving sprites)
        if T > 1:
            obs_diff = (obs[:, 1:] - obs[:, :-1]).abs()
            fg_mask = (obs_diff > self.fg_recon_threshold).float()
            recon_err = (recon[:, 1:] - obs[:, 1:]).pow(2)
            l1_fg = (recon_err * fg_mask).sum() / (fg_mask.sum().clamp(min=1.0))
            l1_bg = (recon_err * (1.0 - fg_mask)).sum() / ((1.0 - fg_mask).sum().clamp(min=1.0))
            
        else:
            l1_fg = torch.tensor(0.0, device=obs.device)
            l1_bg = torch.tensor(0.0, device=obs.device)
        
        # Sprite slot prediction
        if self.sprite_head is not None and sprite_alive is not None and sprite_xy is not None:
            feat_flat = feat.reshape(B*T, -1)
            sprite_out = self.sprite_head(feat_flat).reshape(B, T, self.sprite_slots, 3)
            pred_alive_logits = sprite_out[..., 0]
            pred_xy = torch.sigmoid(sprite_out[..., 1:3])
            true_alive = sprite_alive.float()
            true_xy = sprite_xy.float()
            norm = torch.tensor([
                max(self.obs_shape[2] - 1, 1),
                max(self.obs_shape[1] - 1, 1)
            ], device=obs.device).view(1, 1, 1, 2)
            true_xy_norm = true_xy / norm
            loss_sprite_alive = F.binary_cross_entropy_with_logits(pred_alive_logits, true_alive)
            alive_mask = true_alive.unsqueeze(-1)
            loss_sprite_xy = (alive_mask * F.smooth_l1_loss(pred_xy, true_xy_norm, reduction='none')).sum()
            loss_sprite_xy = loss_sprite_xy / (alive_mask.sum().clamp(min=1.0))
        else:
            loss_sprite_alive = torch.tensor(0.0, device=obs.device)
            loss_sprite_xy = torch.tensor(0.0, device=obs.device)

        # YOLO-style slot head (trained separately)
        if self.yolo_head is not None and sprite_alive is not None and sprite_xy is not None and mario_xy is not None:
            feat_flat = feat.reshape(B*T, -1).detach()
            yolo_out = self.yolo_head(feat_flat).reshape(B, T, self.yolo_slots, 5)
            pred_obj_logits = yolo_out[..., 0]
            pred_xywh = torch.sigmoid(yolo_out[..., 1:5])

            obs_w = max(self.obs_shape[2] - 1, 1)
            obs_h = max(self.obs_shape[1] - 1, 1)
            x_scale = obs_w / float(max(256 - 1, 1))
            y_scale = obs_h / float(max(224 - 1, 1))
            center_x = (sprite_xy[..., 0] + 8.0) * x_scale
            center_y = (sprite_xy[..., 1] + 8.0) * y_scale
            sprite_center = torch.stack([center_x, center_y], dim=-1)
            sprite_wh = torch.tensor([16.0 * x_scale, 16.0 * y_scale], device=obs.device).view(1, 1, 1, 2)

            mario_center = torch.stack([
                (mario_xy[..., 0] + 8.0) * x_scale,
                (mario_xy[..., 1] + 16.0) * y_scale
            ], dim=-1)
            mario_wh = torch.tensor([16.0 * x_scale, 32.0 * y_scale], device=obs.device).view(1, 1, 1, 2)

            norm = torch.tensor([obs_w, obs_h], device=obs.device).view(1, 1, 1, 2)
            sprite_center_norm = sprite_center / norm
            sprite_wh_norm = sprite_wh / norm
            mario_center_norm = mario_center / norm
            mario_wh_norm = mario_wh / norm

            yolo_obj = torch.zeros((B, T, self.yolo_slots), device=obs.device)
            yolo_xywh = torch.zeros((B, T, self.yolo_slots, 4), device=obs.device)
            yolo_obj[..., : self.sprite_slots] = sprite_alive
            yolo_xywh[..., : self.sprite_slots, 0:2] = sprite_center_norm
            yolo_xywh[..., : self.sprite_slots, 2:4] = sprite_wh_norm
            yolo_obj[..., -1] = 1.0
            yolo_xywh[..., -1, 0:2] = mario_center_norm
            yolo_xywh[..., -1, 2:4] = mario_wh_norm

            loss_yolo_obj = F.binary_cross_entropy_with_logits(pred_obj_logits, yolo_obj)
            yolo_mask = yolo_obj.unsqueeze(-1)
            loss_yolo_xywh = (yolo_mask * F.smooth_l1_loss(pred_xywh, yolo_xywh, reduction='none')).sum()
            loss_yolo_xywh = loss_yolo_xywh / (yolo_mask.sum().clamp(min=1.0))
            loss_yolo = loss_yolo_obj + loss_yolo_xywh
        else:
            loss_yolo = torch.tensor(0.0, device=obs.device)

        # Mask head + Dice/BCE loss (uses GT sprite+Mario boxes)
        if self.mask_head is not None and sprite_alive is not None and sprite_xy is not None and mario_xy is not None:
            feat_flat = feat.reshape(B*T, -1)
            mask_logits = self.mask_head(feat_flat).reshape(B, T, 1, self.obs_shape[1], self.obs_shape[2])
            mask_pred = torch.sigmoid(mask_logits)
            target_mask = build_box_mask(sprite_alive, sprite_xy, mario_xy, self.sprite_slots, self.obs_shape, obs.device)
            loss_mask_bce = mask_bce_loss(mask_pred, target_mask)
            loss_mask_dice = dice_loss(mask_pred, target_mask)
            loss_mask = self.mask_bce_weight * loss_mask_bce + self.mask_dice_weight * loss_mask_dice
            mask_iou_value = mask_iou_metric(mask_pred, target_mask)
        else:
            loss_mask = torch.tensor(0.0, device=obs.device)
            loss_mask_bce = torch.tensor(0.0, device=obs.device)
            loss_mask_dice = torch.tensor(0.0, device=obs.device)
            mask_iou_value = torch.tensor(0.0, device=obs.device)

        # Reward
        rew_dist = self.reward_predictor(feat.reshape(B*T, -1))
        loss_rew = -rew_dist.log_prob(symlog(rewards.reshape(B*T))).mean()
        
        # Continue
        cont_pred = self.continue_predictor(feat.reshape(B*T, -1))
        loss_cont = F.binary_cross_entropy(cont_pred.reshape(B, T), 1.0 - terminals)
        
        # KL Loss (KL Balancing)
        # KL(post || prior)
        kl_loss = self.compute_kl_loss(logits, prior_logits)
        
        loss_model = (
            self.recon_weight * loss_recon +
            self.reward_weight * loss_rew +
            self.continue_weight * loss_cont +
            self.kl_scale * kl_loss +
            self.delta_recon_weight * loss_recon_delta +
            self.edge_recon_weight * loss_recon_edge +
            self.fg_recon_weight * l1_fg +
            self.sprite_alive_weight * loss_sprite_alive +
            self.sprite_xy_weight * loss_sprite_xy +
            loss_mask
        )
        
        self.model_opt.zero_grad()
        loss_model.backward()
        self.model_opt.step()

        if self.yolo_opt is not None and loss_yolo.requires_grad:
            self.yolo_opt.zero_grad()
            loss_yolo.backward()
            self.yolo_opt.step()
        
        # 2. Actor-Critic Training (Imagination)
        # Start from the posterior states discovered during WM training
        # Usually we detach the features to not backprop into WM
        start_feat = feat.detach().reshape(B*T, -1)
        # Run imagination for horizon H
        hor = self.imagination_horizon
        img_feats = []
        curr_state = {
            'deter': deter.detach().reshape(B*T, -1),
            'stoch': stoch.detach().reshape(B*T, 32, 32)
        }
        if cell is not None:
            curr_state['cell'] = cell.detach().reshape(B*T, -1)
        
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
        # Maximize returns and entropy
        actor_entropy = action_dist.entropy().mean()
        loss_actor = -returns.mean() - self.entropy_scale * actor_entropy
        
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
            'loss_recon_delta': loss_recon_delta.item(),
            'loss_recon_edge': loss_recon_edge.item(),
            'loss_recon_fg': l1_fg.item(),
            'loss_recon_bg': l1_bg.item(),
            'loss_sprite_alive': loss_sprite_alive.item(),
            'loss_sprite_xy': loss_sprite_xy.item(),
            'loss_yolo': loss_yolo.item(),
            'loss_mask': loss_mask.item(),
            'loss_mask_bce': loss_mask_bce.item(),
            'loss_mask_dice': loss_mask_dice.item(),
            'mask_iou': mask_iou_value.item(),
            'loss_rew': loss_rew.item(),
            'loss_kl': kl_loss.item(),
            'loss_actor': loss_actor.item(),
            'actor_entropy': actor_entropy.item(),
            'actor_entropy_reg': (self.entropy_scale * actor_entropy).item(),
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
        if self.free_nats > 0.0:
            free_nats = torch.as_tensor(self.free_nats, device=kl_post.device)
            kl_post = torch.maximum(kl_post, free_nats)
            kl_prior = torch.maximum(kl_prior, free_nats)
        
        return 0.2 * kl_post.mean() + 0.8 * kl_prior.mean()

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
