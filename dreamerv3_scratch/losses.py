import torch
import torch.nn.functional as F


def build_box_mask(sprite_alive, sprite_xy, mario_xy, sprite_slots, obs_shape, device):
    """Build a binary mask from sprite and Mario boxes in observation space."""
    B, T = sprite_alive.shape[:2]
    mask = torch.zeros((B, T, 1, obs_shape[1], obs_shape[2]), device=device)
    obs_w = max(obs_shape[2] - 1, 1)
    obs_h = max(obs_shape[1] - 1, 1)
    x_scale = obs_w / float(max(256 - 1, 1))
    y_scale = obs_h / float(max(224 - 1, 1))
    for b in range(B):
        for t in range(T):
            alive = sprite_alive[b, t]
            coords = sprite_xy[b, t]
            for s in range(sprite_slots):
                if alive[s] > 0.5:
                    x0 = int(coords[s, 0] * x_scale)
                    y0 = int(coords[s, 1] * y_scale)
                    x1 = int((coords[s, 0] + 16.0) * x_scale)
                    y1 = int((coords[s, 1] + 16.0) * y_scale)
                    x0 = max(min(x0, obs_w), 0)
                    x1 = max(min(x1, obs_w), 0)
                    y0 = max(min(y0, obs_h), 0)
                    y1 = max(min(y1, obs_h), 0)
                    mask[b, t, 0, y0:y1 + 1, x0:x1 + 1] = 1.0
            mx = mario_xy[b, t, 0]
            my = mario_xy[b, t, 1]
            x0 = int(mx * x_scale)
            y0 = int(my * y_scale)
            x1 = int((mx + 16.0) * x_scale)
            y1 = int((my + 32.0) * y_scale)
            x0 = max(min(x0, obs_w), 0)
            x1 = max(min(x1, obs_w), 0)
            y0 = max(min(y0, obs_h), 0)
            y1 = max(min(y1, obs_h), 0)
            mask[b, t, 0, y0:y1 + 1, x0:x1 + 1] = 1.0
    return mask


def dice_loss(pred, target, eps=1e-6):
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def mask_iou(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum(dim=(2, 3, 4))
    union = pred_bin.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def mask_bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target)