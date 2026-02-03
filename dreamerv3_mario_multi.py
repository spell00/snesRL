import sys
import os
import torch
import numpy as np
import time
import functools
import cv2

# Add local folders to path
sys.path.append(os.path.join(os.getcwd(), 'dreamerv3_scratch'))
sys.path.append(os.path.join(os.getcwd(), 'zelda_gym'))

from zelda_gym.mario_bizhawk_env import MarioBizHawkEnv
from dreamerv3_scratch.agent import DreamerAgent
from dreamerv3_scratch.buffer import ReplayBuffer
from dreamerv3_scratch.parallel_envs import ParallelEnvs

import argparse
import multiprocessing as mp
import subprocess
import shlex
from tensorboardX import SummaryWriter

# --- Global cleanup for zombie processes ---
import atexit
import signal


def normalize_obs(obs, device):
    if isinstance(obs, torch.Tensor):
        return obs.float().to(device) / 255.0
    return torch.from_numpy(obs).float().to(device) / 255.0

def _draw_sprites(frame, sprite_alive, sprite_xy, color=(0, 255, 0), radius=2):
    if frame is None or sprite_alive is None or sprite_xy is None:
        return frame
    height, width = frame.shape[:2]
    x_scale = width / 256.0
    y_scale = height / 224.0
    for i in range(len(sprite_alive)):
        if sprite_alive[i] > 0.5:
            x = int(sprite_xy[i][0] * x_scale)
            y = int(sprite_xy[i][1] * y_scale)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), radius, color, -1)
    return frame

def _draw_pred_sprites(frame, pred_alive, pred_xy, color=(0, 255, 0), radius=2):
    if frame is None or pred_alive is None or pred_xy is None:
        return frame
    height, width = frame.shape[:2]
    for i in range(len(pred_alive)):
        if pred_alive[i] > 0.5:
            x = int(pred_xy[i][0] * (width - 1))
            y = int(pred_xy[i][1] * (height - 1))
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), radius, color, -1)
    return frame

def _draw_boxes(frame, positions, color=(255, 0, 0), half_size=2):
    if frame is None:
        return frame
    height, width = frame.shape[:2]
    for x, y in positions:
        x0 = max(int(x - half_size), 0)
        y0 = max(int(y - half_size), 0)
        x1 = min(int(x + half_size), width - 1)
        y1 = min(int(y + half_size), height - 1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
    return frame

def _draw_box_xywh(frame, center_x, center_y, width, height, color=(0, 255, 0)):
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    half_w = width / 2.0
    half_h = height / 2.0
    x0 = max(int(center_x - half_w), 0)
    y0 = max(int(center_y - half_h), 0)
    x1 = min(int(center_x + half_w), w - 1)
    y1 = min(int(center_y + half_h), h - 1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
    return frame

def _masked_boxes(frame, positions, half_size=2):
    if frame is None:
        return frame
    height, width = frame.shape[:2]
    masked = np.zeros_like(frame)
    for x, y in positions:
        x0 = max(int(x - half_size), 0)
        y0 = max(int(y - half_size), 0)
        x1 = min(int(x + half_size), width - 1)
        y1 = min(int(y + half_size), height - 1)
        masked[y0:y1 + 1, x0:x1 + 1] = frame[y0:y1 + 1, x0:x1 + 1]
    return masked

def _scale_coords(sprite_xy, src_width, src_height, dst_width, dst_height):
    x_scale = (dst_width - 1) / float(max(src_width - 1, 1))
    y_scale = (dst_height - 1) / float(max(src_height - 1, 1))
    scaled = []
    for x, y in sprite_xy:
        scaled.append((x * x_scale, y * y_scale))
    return scaled

def kill_all_bizhawk_related():
    for proc in ["EmuHawk", "mono", "Xvfb"]:
        try:
            subprocess.run(["pkill", "-9", "-f", proc], check=False)
        except Exception:
            pass


def cleanup_pulseaudio():
    if not sys.platform.startswith("linux"):
        return
    env = os.environ.copy()
    try:
        subprocess.run(["pactl", "unload-module", "module-null-sink"], env=env, check=False)
    except Exception:
        pass
    try:
        subprocess.run(["pkill", "-9", "-f", "pulseaudio"], check=False)
    except Exception:
        pass


def ensure_pulseaudio_nullsink():
    """Starts PulseAudio and loads a null sink once for the entire session."""
    cleanup_pulseaudio()
    print("--- Initializing Global PulseAudio Null Sink ---")
    env = os.environ.copy()
    # Start PulseAudio if not running
    subprocess.run(["pulseaudio", "--start", "--exit-idle-time=-1"], env=env, check=False)
    # Load null-sink. pactl returns 0 if success or if already loaded (usually)
    # We don't check=True because it might already be loaded
    subprocess.run(
        ["pactl", "load-module", "module-null-sink", 
         "sink_name=DummySink", "sink_properties=device.description=DummySink"],
        env=env, check=False
    )
    # Set global env vars for workers to inherit
    os.environ["PULSE_SINK"] = "DummySink"
    os.environ["ALSOFT_DRIVERS"] = "pulse"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel training environments")
    parser.add_argument("--total_steps", type=int, default=2000000, help="Total training steps")
    parser.add_argument("--frameskip", type=int, default=4, help="Frameskip for the environment")
    parser.add_argument("--obs_size", type=int, default=128, help="Observation size")
    parser.add_argument("--death_penalty", type=float, default=-5.0, help="Penalty for dying")
    parser.add_argument("--exploration_bonus", type=float, default=10, help="Exploration bonus amount")
    parser.add_argument("--enable_cell_exploration", action="store_true", default=False, help="Enable cell-based exploration")
    parser.add_argument("--level", type=int, default=3, help="Level index: 0=random, 1-N=specific savestate")
    parser.add_argument("--no_load", action="store_true", help="Do not load the latest checkpoint")
    parser.add_argument("--headless", type=int, default=1, help="Run headless")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kl_scale", type=float, default=1.0, help="KL loss scale")
    parser.add_argument("--free_nats", type=float, default=1.0, help="Free nats for KL clamp")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level: 0=silent, 1=default, 2=debug prints")
    parser.add_argument("--screenshot_every", type=int, default=1, help="Capture a screenshot every N steps")
    parser.add_argument("--delta_recon_weight", type=float, default=0.0, help="Delta-frame recon loss weight")
    parser.add_argument("--edge_recon_weight", type=float, default=0.0, help="Edge recon loss weight")
    parser.add_argument("--fill_buffer_steps", type=int, default=2000, help="Warmup steps before training updates")
    parser.add_argument("--dump_obs_once", action="store_true", help="Dump one raw obs frame to debug_obs_raw.png")
    parser.add_argument("--fg_recon_weight", type=float, default=1.0, help="Foreground recon loss weight")
    parser.add_argument("--fg_recon_threshold", type=float, default=0.1, help="Foreground mask threshold in [0,1]")
    parser.add_argument("--sprite_slots", type=int, default=12, help="Number of sprite slots")
    parser.add_argument("--sprite_alive_weight", type=float, default=1.0, help="Sprite alive loss weight")
    parser.add_argument("--sprite_xy_weight", type=float, default=0.5, help="Sprite xy loss weight")
    parser.add_argument("--box_recon_weight", type=float, default=1.0, help="Recon weight multiplier inside sprite/Mario boxes")
    parser.add_argument("--recon_weight", type=float, default=1.0, help="Recon loss weight")
    parser.add_argument("--reward_weight", type=float, default=1.0, help="Reward loss weight")
    parser.add_argument("--reward_progress_scale", type=float, default=0.01, help="Scale for max progress reward")
    parser.add_argument("--continue_weight", type=float, default=1.0, help="Continue loss weight")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="KL loss weight multiplier")
    parser.add_argument("--buffer_capacity", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--batch_length", type=int, default=64, help="Sequence length for training batches (horizon)")
    parser.add_argument(
        "--rssm_type",
        type=str,
        default="categorical",
        choices=["categorical", "gaussian"],
        help="RSSM Latent Type",
    )
    parser.add_argument(
        "--recurrent_model",
        type=str,
        default="rssm",
        choices=["rssm", "gru", "lstm"],
        help="Recurrent world model: rssm (default), gru, or lstm",
    )
    return parser.parse_args()

def make_env(rank, args):
    env = MarioBizHawkEnv(
        rank=rank,
        headless=args.headless,
        obs_size=args.obs_size,
        sprite_slots=args.sprite_slots,
        action_type="discrete",
        frameskip=args.frameskip,
        screenshot_every=args.screenshot_every,
        return_full_res=(rank == 1),
        keep_screenshots=False,
        death_penalty=args.death_penalty,
        exploration_bonus=args.exploration_bonus,
        enable_cell_exploration=args.enable_cell_exploration,
        progress_scale=getattr(args, 'reward_progress_scale', 0.01),
        # startup_sleep_s=10.0,
        reset_timeout_s=60.0  # Spawned workers need more time
    )
    
    # Wrap reset to handle level selection
    if args.level != 0:
        orig_reset = env.reset
        level_index = args.level # capture current level

        def reset_with_level(*, seed=None, options=None):
            opts = dict(options) if options else {}
            opts["savestate"] = env.get_savestate_by_index(level_index)
            return orig_reset(seed=seed, options=opts)
        
        env.reset = reset_with_level
    
    return env

def latest_checkpoint():
    if not os.path.exists('checkpoints'):
        return None
    ckpts = [f for f in os.listdir('checkpoints') if f.startswith("dreamer_mario_multi_") and f.endswith(".pt")]
    if not ckpts:
        return None
    try:
        # Sort by step number: dreamer_mario_multi_5000.pt -> 5000
        ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join('checkpoints', ckpts[-1])
    except:
        return None

def main():
    # Register cleanup on exit and signals (main process only, before any envs are created)
    # Register cleanup only in the true main process
    import multiprocessing
    if multiprocessing.current_process().name == "MainProcess":
        kill_all_bizhawk_related()
        cleanup_pulseaudio()
        atexit.register(kill_all_bizhawk_related)
        atexit.register(cleanup_pulseaudio)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda signum, frame: (kill_all_bizhawk_related(), cleanup_pulseaudio(), exit(0)))
    args = parse_args()
    n_envs = args.n_envs
    obs_shape = (1, args.obs_size, args.obs_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} with {n_envs} environments")

    # TensorBoard writer
    writer = SummaryWriter(log_dir="logs/tensorboard_multi")

    # Ensure stable audio for all workers
    if sys.platform.startswith("linux"):
        ensure_pulseaudio_nullsink()

    # Force spawn to avoid deadlocks with Torch/OpenCV/X11
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Initialize Parallel Envs
    action_dim = MarioBizHawkEnv.discrete_action_count()

    env_fn = functools.partial(make_env, args=args)
    envs = ParallelEnvs(n_envs, env_fn, obs_shape, action_dim, device)
    
    # Initialize Agent
    agent = DreamerAgent(
        obs_shape,
        action_dim,
        lr=1e-4,
        device=device,
        kl_scale=args.kl_scale * args.kl_weight,
        free_nats=args.free_nats,
        recon_weight=args.recon_weight,
        reward_weight=args.reward_weight,
        continue_weight=args.continue_weight,
        delta_recon_weight=args.delta_recon_weight,
        edge_recon_weight=args.edge_recon_weight,
        fg_recon_weight=args.fg_recon_weight,
        fg_recon_threshold=args.fg_recon_threshold,
        sprite_slots=args.sprite_slots,
        sprite_alive_weight=args.sprite_alive_weight,
        sprite_xy_weight=args.sprite_xy_weight,
        box_recon_weight=args.box_recon_weight,
        mask_bce_weight=1.0,
        mask_dice_weight=1.0,
        recurrent_model=args.recurrent_model,
        rssm_type=args.rssm_type,
    )
    
    # Load latest checkpoint if exists
    if not args.no_load:
        ckpt = latest_checkpoint()
        if ckpt:
            print(f"Loading checkpoint: {ckpt}")
            agent.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print("No checkpoint found. Starting from scratch.")
    else:
        print("Starting FRESH training (--no_load).")

    # Increased sequence length for better memory
    # Increased sequence length for better memory
    buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        sequence_length=args.batch_length,
        obs_shape=obs_shape,
        action_dim=action_dim,
        sprite_slots=args.sprite_slots,
    )
    
    # Training Loop
    total_steps = args.total_steps
    step = 0
    
    obs, infos = envs.reset()
    obs_raw = obs
    assert obs_raw.dtype == np.uint8
    dumped_obs = False
    obs = normalize_obs(obs_raw, device)
    init = agent.rssm.initial_state(n_envs, device)
    states = {k: v.clone() for k, v in init.items()}
    action_onehots = torch.zeros(n_envs, action_dim).to(device)
    print("Starting collection...")
    rewards = []
    reward_components = []
    episode_idx = [0 for _ in range(n_envs)]
    episode_returns = [0.0 for _ in range(n_envs)]
    episode_lengths = [0 for _ in range(n_envs)]
    episode_component_totals = [
        {
            "reward_progress": 0.0,
            "reward_novelty": 0.0,
            "reward_cell": 0.0,
            "reward_win": 0.0,
            "reward_death": 0.0,
            "reward_stuck": 0.0,
        }
        for _ in range(n_envs)
    ]
    frames_per_env = [[] for _ in range(n_envs)]
    resized_frames_per_env = [[] for _ in range(n_envs)]
    recon_frames_per_env = [[] for _ in range(n_envs)]
    sprite_frames_per_env = [[] for _ in range(n_envs)]
    recon_combined_frames_per_env = [[] for _ in range(n_envs)]
    overlay_frames_per_env = [[] for _ in range(n_envs)]
    masked_frames_per_env = [[] for _ in range(n_envs)]
    yolo_frames_per_env = [[] for _ in range(n_envs)]
    while step < total_steps:
        # Act
        # agent.act usually expects (1, C, H, W). We need to batch it.
        # Let's use the underlying modules for batch inference
        # obs = normalize_obs(obs, device)
        with torch.no_grad():
            embeds = agent.encoder(obs)
            states = agent.rssm.observe(embeds, action_onehots, states)
            feats = torch.cat([states['deter'], states['stoch'].reshape(n_envs, -1)], dim=-1)
            action_dists = agent.actor(feats)
            actions = action_dists.sample()
            action_onehots = torch.nn.functional.one_hot(actions, action_dim).float()
            recon_batch = agent.decoder(feats).sigmoid()
            sprite_pred_alive = None
            sprite_pred_xy = None
            if agent.sprite_head is not None:
                sprite_out = agent.sprite_head(feats).reshape(n_envs, agent.sprite_slots, 3)
                sprite_pred_alive = torch.sigmoid(sprite_out[..., 0])
                sprite_pred_xy = torch.sigmoid(sprite_out[..., 1:3])
            yolo_pred_obj = None
            yolo_pred_xywh = None
            if agent.yolo_head is not None:
                yolo_out = agent.yolo_head(feats).reshape(n_envs, agent.yolo_slots, 5)
                yolo_pred_obj = torch.sigmoid(yolo_out[..., 0])
                yolo_pred_xywh = torch.sigmoid(yolo_out[..., 1:5])
            
        actions_np = actions.cpu().numpy()
        
        # Step Envs
        next_obs, reward, terms, truncs, infos = envs.step(actions_np)
        next_obs_raw = next_obs
        assert next_obs_raw.dtype == np.uint8
        next_obs = normalize_obs(next_obs_raw, device)
        
        step_component_sum = {
            "reward_progress": 0.0,
            "reward_novelty": 0.0,
            "reward_cell": 0.0,
            "reward_win": 0.0,
            "reward_death": 0.0,
            "reward_stuck": 0.0,
        }
        for i in range(n_envs):
            full_res = infos[i].get("full_res_frame") if isinstance(infos[i], dict) else None
            if full_res is not None:
                frames_per_env[i].append(full_res)
            # Save resized frame to keep cadence aligned with full-res
            frame = next_obs_raw[i]
            if frame.ndim == 3 and frame.shape[0] in (1, 3):
                frame = np.transpose(frame, (1, 2, 0))
            if frame.ndim == 2:
                frame = frame[:, :, None]
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=2)
            resized_frames_per_env[i].append(frame)
            recon = recon_batch[i].detach().cpu().numpy()
            if recon.ndim == 3 and recon.shape[0] in (1, 3):
                recon = np.transpose(recon, (1, 2, 0))
            if recon.ndim == 2:
                recon = recon[:, :, None]
            if recon.shape[-1] == 1:
                recon = np.repeat(recon, 3, axis=2)
            recon = (recon * 255.0).astype(np.uint8)
            recon_frames_per_env[i].append(recon)
            sprite_alive = infos[i].get("sprite_alive") if isinstance(infos[i], dict) else None
            sprite_xy = infos[i].get("sprite_xy") if isinstance(infos[i], dict) else None
            if args.verbose >= 2 and (len(recon_frames_per_env[i]) == 1 or len(recon_frames_per_env[i]) % 60 == 0):
                recon_stats = recon_batch[i]
                mario_x = infos[i].get("mario_screen_x") if isinstance(infos[i], dict) else None
                mario_y = infos[i].get("mario_screen_y") if isinstance(infos[i], dict) else None
                active_count = int(np.sum(sprite_alive > 0.5)) if sprite_alive is not None else 0
                active_coords = []
                if sprite_alive is not None and sprite_xy is not None:
                    active_indices = np.where(sprite_alive > 0.5)[0].tolist()
                    for idx in active_indices[:5]:
                        active_coords.append((int(idx), float(sprite_xy[idx][0]), float(sprite_xy[idx][1])))
                print(
                    f"[Recon][env{i}] min/max/mean: "
                    f"{recon_stats.min().item():.4f}/"
                    f"{recon_stats.max().item():.4f}/"
                    f"{recon_stats.mean().item():.4f} | "
                    f"Mario: ({mario_x},{mario_y}) | "
                    f"sprites={active_count} top5={active_coords}"
                )
            if sprite_pred_alive is not None and sprite_pred_xy is not None:
                pred_alive = sprite_pred_alive[i].detach().cpu().numpy()
                pred_xy = sprite_pred_xy[i].detach().cpu().numpy()
                if args.verbose >= 2 and (len(sprite_frames_per_env[i]) == 0 or len(sprite_frames_per_env[i]) % 60 == 0):
                    max_alive = float(pred_alive.max()) if pred_alive.size else 0.0
                    mean_alive = float(pred_alive.mean()) if pred_alive.size else 0.0
                    above_thresh = int((pred_alive > 0.5).sum()) if pred_alive.size else 0
                    print(f"[PredSprites][env{i}] alive max={max_alive:.3f} mean={mean_alive:.3f} >0.5={above_thresh}")
                combined_frame = _draw_pred_sprites(recon.copy(), pred_alive, pred_xy, color=(0, 255, 0))
                recon_combined_frames_per_env[i].append(combined_frame)
            overlay_frame = frame.copy()
            height, width = overlay_frame.shape[:2]
            active_indices = np.where(sprite_alive > 0.5)[0].tolist() if sprite_alive is not None else []
            sprite_coords = []
            if sprite_xy is not None:
                sprite_coords = [(float(sprite_xy[idx][0]) + 8.0, float(sprite_xy[idx][1]) + 8.0) for idx in active_indices]
            scaled_sprite_coords = _scale_coords(sprite_coords, 256.0, 224.0, width, height)
            if scaled_sprite_coords:
                overlay_frame = _draw_boxes(overlay_frame, scaled_sprite_coords, color=(0, 255, 0), half_size=4)
            mario_x = infos[i].get("mario_screen_x") if isinstance(infos[i], dict) else None
            mario_y = infos[i].get("mario_screen_y") if isinstance(infos[i], dict) else None
            mario_scaled = []
            if mario_x is not None and mario_y is not None:
                mario_scaled = _scale_coords([(float(mario_x) + 8.0, float(mario_y) + 16.0)], 256.0, 224.0, width, height)
                overlay_frame = _draw_boxes(overlay_frame, mario_scaled, color=(255, 0, 0), half_size=12)
            overlay_frames_per_env[i].append(overlay_frame)
            masked_positions = list(scaled_sprite_coords)
            if mario_scaled:
                masked_positions += mario_scaled
            if masked_positions:
                masked_frame = _masked_boxes(frame, masked_positions, half_size=4)
            else:
                masked_frame = frame.copy()
            if mario_scaled:
                masked_frame = _masked_boxes(masked_frame, mario_scaled, half_size=12)
            masked_frames_per_env[i].append(masked_frame)
            sprite_masked = _masked_boxes(recon, scaled_sprite_coords, half_size=4) if scaled_sprite_coords else np.zeros_like(recon)
            if mario_scaled:
                mario_mask = _masked_boxes(recon, mario_scaled, half_size=12)
                mario_pixels = np.any(mario_mask != 0, axis=-1)
                sprite_masked[mario_pixels] = mario_mask[mario_pixels]
            sprite_frames_per_env[i].append(sprite_masked)
            if args.verbose >= 2 and (step % (n_envs * 10) < n_envs):
                print(f"[Overlay][env{i}] mario=({mario_x},{mario_y}) sprites={len(active_indices)}")
            if yolo_pred_obj is not None and yolo_pred_xywh is not None and i == 1:
                yolo_frame = frame.copy()
                height, width = yolo_frame.shape[:2]
                pred_obj = yolo_pred_obj[i].detach().cpu().numpy()
                pred_xywh = yolo_pred_xywh[i].detach().cpu().numpy()
                # Draw ground-truth sprite and Mario boxes in black
                if isinstance(infos[i], dict):
                    gt_alive = infos[i].get("sprite_alive")
                    gt_xy = infos[i].get("sprite_xy")
                    if gt_alive is not None and gt_xy is not None:
                        gt_alive = np.array(gt_alive)
                        gt_xy = np.array(gt_xy)
                        for slot_idx in np.where(gt_alive > 0.5)[0].tolist():
                            cx = (gt_xy[slot_idx][0] + 8.0) * (width - 1) / 255.0
                            cy = (gt_xy[slot_idx][1] + 8.0) * (height - 1) / 223.0
                            bw = 16.0 * (width - 1) / 255.0
                            bh = 16.0 * (height - 1) / 223.0
                            yolo_frame = _draw_box_xywh(yolo_frame, cx, cy, bw, bh, color=(0, 0, 0))
                    mario_x = infos[i].get("mario_screen_x")
                    mario_y = infos[i].get("mario_screen_y")
                    if mario_x is not None and mario_y is not None:
                        cx = (float(mario_x) + 8.0) * (width - 1) / 255.0
                        cy = (float(mario_y) + 16.0) * (height - 1) / 223.0
                        bw = 16.0 * (width - 1) / 255.0
                        bh = 32.0 * (height - 1) / 223.0
                        yolo_frame = _draw_box_xywh(yolo_frame, cx, cy, bw, bh, color=(0, 0, 0))
                for slot_idx in range(pred_xywh.shape[0]):
                    if pred_obj[slot_idx] < 0.5:
                        continue
                    cx = pred_xywh[slot_idx][0] * (width - 1)
                    cy = pred_xywh[slot_idx][1] * (height - 1)
                    bw = pred_xywh[slot_idx][2] * (width - 1)
                    bh = pred_xywh[slot_idx][3] * (height - 1)
                    color = (0, 255, 0) if slot_idx == pred_xywh.shape[0] - 1 else (0, 0, 255)
                    yolo_frame = _draw_box_xywh(yolo_frame, cx, cy, bw, bh, color=color)
                yolo_frames_per_env[i].append(yolo_frame)
            sprite_alive = infos[i].get("sprite_alive") if isinstance(infos[i], dict) else None
            sprite_xy = infos[i].get("sprite_xy") if isinstance(infos[i], dict) else None
            mario_xy = None
            if isinstance(infos[i], dict):
                mario_x = infos[i].get("mario_screen_x")
                mario_y = infos[i].get("mario_screen_y")
                if mario_x is not None and mario_y is not None:
                    mario_xy = np.array([mario_x, mario_y], dtype=np.float32)
            buffer.add(
                obs_raw[i],
                action_onehots[i],
                reward[i],
                terms[i] or truncs[i],
                sprite_alive=sprite_alive,
                sprite_xy=sprite_xy,
                mario_xy=mario_xy,
            )
            episode_returns[i] += float(reward[i])
            episode_lengths[i] += 1
            if isinstance(infos[i], dict):
                for key in step_component_sum:
                    comp_val = float(infos[i].get(key, 0.0))
                    step_component_sum[key] += comp_val
                    episode_component_totals[i][key] += comp_val
            if args.dump_obs_once and not dumped_obs:
                frame = next_obs_raw[i]
                if frame.ndim == 3 and frame.shape[0] in (1, 3):
                    frame = np.transpose(frame, (1, 2, 0))
                if frame.ndim == 2:
                    frame = frame[:, :, None]
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=2)
                cv2.imwrite("debug_obs_raw.png", frame)
                print("Wrote debug_obs_raw.png")
                dumped_obs = True
        obs_raw = next_obs_raw
        obs = next_obs
        step += n_envs
        reward_components.append(step_component_sum)
        # Reset RSSM state and save video for finished episodes
        for i in range(n_envs):
            if terms[i] or truncs[i]:
                if isinstance(infos[i], dict):
                    end_reason = infos[i].get("termination_reason") or infos[i].get("end_reason")
                    death_flag = infos[i].get("death")
                else:
                    end_reason = None
                    death_flag = None
                if terms[i]:
                    status = "terminated"
                else:
                    status = "truncated"
                reason_msg = f" reason={end_reason}" if end_reason is not None else ""
                death_msg = f" death={death_flag}" if death_flag is not None else ""
                print(f"[EpisodeEnd][env{i}] {status}{reason_msg}{death_msg}")
                # Save videos for this env
                if i == 1 and frames_per_env[i] and resized_frames_per_env[i] and recon_frames_per_env[i]:
                    os.makedirs('videos', exist_ok=True)
                    video_path = os.path.join('videos', f'env{i}_episode_{episode_idx[i]:05d}.mp4')
                    height, width = resized_frames_per_env[i][0].shape[:2]
                    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                    for f in resized_frames_per_env[i]:
                        out.write(f)
                    out.release()
                    full_video_path = os.path.join('videos', f'env{i}_episode_{episode_idx[i]:05d}_full.mp4')
                    full_height, full_width = frames_per_env[i][0].shape[:2]
                    out_full = cv2.VideoWriter(full_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (full_width, full_height))
                    for f in frames_per_env[i]:
                        out_full.write(f)
                    out_full.release()
                    recon_video_path = os.path.join('videos', f'env{i}_episode_{episode_idx[i]:05d}_recon.mp4')
                    recon_height, recon_width = recon_frames_per_env[i][0].shape[:2]
                    out_recon = cv2.VideoWriter(recon_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (recon_width, recon_height))
                    for f in recon_frames_per_env[i]:
                        out_recon.write(f)
                    out_recon.release()
                    if sprite_frames_per_env[i]:
                        sprite_video_path = os.path.join('videos', f'env{i}_episode_{episode_idx[i]:05d}_recon_sprites.mp4')
                        sprite_height, sprite_width = sprite_frames_per_env[i][0].shape[:2]
                        out_sprite = cv2.VideoWriter(sprite_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (sprite_width, sprite_height))
                        for f in sprite_frames_per_env[i]:
                            out_sprite.write(f)
                        out_sprite.release()
                    if recon_combined_frames_per_env[i]:
                        combined_video_path = os.path.join('videos', f'env{i}_episode_{episode_idx[i]:05d}_recon_combined.mp4')
                        combined_height, combined_width = recon_combined_frames_per_env[i][0].shape[:2]
                        out_combined = cv2.VideoWriter(combined_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (combined_width, combined_height))
                        for f in recon_combined_frames_per_env[i]:
                            out_combined.write(f)
                        out_combined.release()
                    if overlay_frames_per_env[i]:
                        overlay_video_path = os.path.join('videos', f'env{i}_episode_{episode_idx[i]:05d}_overlay.mp4')
                        overlay_height, overlay_width = overlay_frames_per_env[i][0].shape[:2]
                        out_overlay = cv2.VideoWriter(overlay_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (overlay_width, overlay_height))
                        for f in overlay_frames_per_env[i]:
                            out_overlay.write(f)
                        out_overlay.release()
                    if masked_frames_per_env[i]:
                        masked_video_path = os.path.join('videos', f'env{i}_episode_{episode_idx[i]:05d}_masked.mp4')
                        masked_height, masked_width = masked_frames_per_env[i][0].shape[:2]
                        out_masked = cv2.VideoWriter(masked_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (masked_width, masked_height))
                        for f in masked_frames_per_env[i]:
                            out_masked.write(f)
                        out_masked.release()
                    if yolo_frames_per_env[i]:
                        yolo_video_path = os.path.join('videos', f'env{i}_episode_{episode_idx[i]:05d}_yolo_overlay.mp4')
                        yolo_height, yolo_width = yolo_frames_per_env[i][0].shape[:2]
                        out_yolo = cv2.VideoWriter(yolo_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (yolo_width, yolo_height))
                        for f in yolo_frames_per_env[i]:
                            out_yolo.write(f)
                        out_yolo.release()
                    num_frames = len(frames_per_env[i])
                    print(f"Saved video: {video_path} ({num_frames} frames)")
                    print(f"Saved full-res video: {full_video_path} ({num_frames} frames)")
                    print(f"Saved recon video: {recon_video_path} ({num_frames} frames)")
                    if sprite_frames_per_env[i]:
                        print(f"Saved sprite recon video: {sprite_video_path} ({num_frames} frames)")
                    if recon_combined_frames_per_env[i]:
                        print(f"Saved combined recon video: {combined_video_path} ({num_frames} frames)")
                    if overlay_frames_per_env[i]:
                        print(f"Saved overlay video: {overlay_video_path} ({num_frames} frames)")
                    if masked_frames_per_env[i]:
                        print(f"Saved masked video: {masked_video_path} ({num_frames} frames)")
                    if yolo_frames_per_env[i]:
                        print(f"Saved yolo overlay video: {yolo_video_path} ({num_frames} frames)")
                    writer.add_scalar("Reward/EpisodeTotal", episode_returns[i], step)
                    writer.add_scalar("Reward/EpisodeLength", episode_lengths[i], step)
                    for key, total in episode_component_totals[i].items():
                        writer.add_scalar(f"Reward/Episode{key.replace('reward_', '').title()}", total, step)
                    episode_idx[i] += 1
                    frames_per_env[i] = []
                    resized_frames_per_env[i] = []
                    recon_frames_per_env[i] = []
                    sprite_frames_per_env[i] = []
                    recon_combined_frames_per_env[i] = []
                    overlay_frames_per_env[i] = []
                    masked_frames_per_env[i] = []
                    yolo_frames_per_env[i] = []
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0
                    for key in episode_component_totals[i]:
                        episode_component_totals[i][key] = 0.0
                # Zero out state for this env
                states['deter'][i].zero_()
                states['stoch'][i].zero_()
                if 'cell' in states:
                    states['cell'][i].zero_()
                action_onehots[i].zero_()

        # Train
        reward_per_env = reward
        reward = np.sum(reward_per_env, axis=0).item()  # Sum rewards across envs if needed
        # Add to buffer and collect frames for video
        if args.verbose >= 2:
            def _filter_info(obj):
                if not isinstance(obj, dict):
                    return obj
                filtered = {}
                for key, val in obj.items():
                    if isinstance(val, (int, float, str, bool)) or val is None:
                        filtered[key] = val
                return filtered
            filtered_infos = [_filter_info(info) for info in infos]
            env_rewards = np.array(reward_per_env).reshape(-1)
            pos_envs = np.where(env_rewards > 0)[0].tolist()
            neg_envs = np.where(env_rewards < 0)[0].tolist()
            if pos_envs:
                pos_info = {idx: filtered_infos[idx] for idx in pos_envs}
                pos_rewards = {idx: float(env_rewards[idx]) for idx in pos_envs}
                print(f"Step {step} | Positive Reward: {pos_rewards} | Info: {pos_info}")
            if neg_envs:
                neg_info = {idx: filtered_infos[idx] for idx in neg_envs}
                neg_rewards = {idx: float(env_rewards[idx]) for idx in neg_envs}
                print(f"Step {step} | Negative Reward: {neg_rewards} | Info: {neg_info}")
        
        if step > args.fill_buffer_steps:
            rewards.append(reward)
            if step % (n_envs * 5) == 0:
                batch = buffer.sample(16)
                metrics = agent.train_step(*[t.to(device) for t in batch])
                
                if step % 1000 < n_envs:
                    # Calculate mean across the collected list of reward-batches
                    avg_reward = np.mean(rewards)
                    print(f"Step {step} | Avg Step Reward: {avg_reward:.4f} | Model Loss: {metrics['loss_model']:.4f} | Actor Loss: {metrics['loss_actor']:.4f}")
                    # Log to TensorBoard
                    writer.add_scalar("Reward/AvgStepReward", avg_reward, step)
                    if reward_components:
                        component_means = {}
                        for key in reward_components[0].keys():
                            component_means[key] = float(np.mean([entry[key] for entry in reward_components]))
                        for key, value in component_means.items():
                            metric_name = key.replace("reward_", "")
                            writer.add_scalar(f"Reward/AvgStep{metric_name.title()}", value, step)
                    writer.add_scalar("Loss/Model", metrics['loss_model'], step)
                    writer.add_scalar("Loss/Recon", metrics['loss_recon'], step)
                    writer.add_scalar("Loss/ReconDelta", metrics.get('loss_recon_delta', 0.0), step)
                    writer.add_scalar("Loss/ReconEdge", metrics.get('loss_recon_edge', 0.0), step)
                    writer.add_scalar("Loss/ReconFg", metrics.get('loss_recon_fg', 0.0), step)
                    writer.add_scalar("Loss/ReconBg", metrics.get('loss_recon_bg', 0.0), step)
                    writer.add_scalar("Loss/SpriteAlive", metrics.get('loss_sprite_alive', 0.0), step)
                    writer.add_scalar("Loss/SpriteXY", metrics.get('loss_sprite_xy', 0.0), step)
                    writer.add_scalar("Loss/YOLO", metrics.get('loss_yolo', 0.0), step)
                    writer.add_scalar("Loss/Mask", metrics.get('loss_mask', 0.0), step)
                    writer.add_scalar("Loss/MaskBCE", metrics.get('loss_mask_bce', 0.0), step)
                    writer.add_scalar("Loss/MaskDice", metrics.get('loss_mask_dice', 0.0), step)
                    writer.add_scalar("Metric/MaskIoU", metrics.get('mask_iou', 0.0), step)
                    writer.add_scalar("Loss/Reward", metrics['loss_rew'], step)
                    writer.add_scalar("Loss/KL", metrics['loss_kl'], step)
                    writer.add_scalar("Loss/Actor", metrics['loss_actor'], step)
                    writer.add_scalar("Loss/Critic", metrics['loss_critic'], step)
                    rewards = [] # Reset after printing/logging
                    reward_components = []
                    
                    if step % 5000 < n_envs:
                        os.makedirs('checkpoints', exist_ok=True)
                        torch.save(agent.state_dict(), f'checkpoints/dreamer_mario_multi_{step}.pt')

    writer.close()

if __name__ == "__main__":
    main()
