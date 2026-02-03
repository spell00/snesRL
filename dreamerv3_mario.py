import sys
import os
import torch
import numpy as np
import cv2
import time
import argparse

# Add local folders to path
sys.path.append(os.path.join(os.getcwd(), 'dreamerv3_scratch'))
sys.path.append(os.path.join(os.getcwd(), 'zelda_gym'))

from mario_bizhawk_env import MarioBizHawkEnv
from agent import DreamerAgent
from buffer import ReplayBuffer

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot_every", type=int, default=1)
    parser.add_argument("--obs_size", type=int, default=128)
    parser.add_argument("--delta_recon_weight", type=float, default=0.0)
    parser.add_argument("--edge_recon_weight", type=float, default=0.0)
    parser.add_argument("--dump_obs_once", action="store_true")
    parser.add_argument("--fg_recon_weight", type=float, default=1.0)
    parser.add_argument("--fg_recon_threshold", type=float, default=0.1)
    parser.add_argument("--sprite_slots", type=int, default=12)
    parser.add_argument("--sprite_alive_weight", type=float, default=1.0)
    parser.add_argument("--sprite_xy_weight", type=float, default=0.5)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--reward_weight", type=float, default=1.0)
    parser.add_argument("--continue_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--free_nats", type=float, default=1.0)
    return parser.parse_args()

def main():
    args = parse_args()
    # Config
    obs_shape = (1, args.obs_size, args.obs_size) # Grayscale
    action_type = "discrete"
    kl_scale = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Env
    env = MarioBizHawkEnv(
        rank=0,
        headless=False,
        obs_size=args.obs_size,
        sprite_slots=args.sprite_slots,
        action_type=action_type,
        frameskip=4,
        screenshot_every=args.screenshot_every,
        return_full_res=True,
        keep_screenshots=False,
    )
    
    action_dim = env.action_space.n
    
    # Initialize Agent
    agent = DreamerAgent(
        obs_shape,
        action_dim,
        device=device,
        kl_scale=kl_scale * args.kl_weight,
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
    )
    buffer = ReplayBuffer(
        capacity=100000,
        sequence_length=32,
        obs_shape=obs_shape,
        action_dim=action_dim,
        sprite_slots=args.sprite_slots,
    )
    
    # Training Loop
    total_steps = 1000000
    step = 0
    
    episode_idx = 0
    dumped_obs = False
    while step < total_steps:
        obs, info = env.reset()
        obs_t = torch.from_numpy(obs).float().to(device) / 255.0
        state = agent.rssm.initial_state(1, device)
        action_onehot = torch.zeros(1, action_dim).to(device)
        done = False
        episode_reward = 0
        frames = []
        full_frames = []
        recon_frames = []
        sprite_frames = []
        recon_combined_frames = []
        while not done:
            # Save frames for both resized and full-res videos
            frame = np.repeat(obs, 3, axis=2) if obs.shape[2] == 1 else obs
            frames.append(frame)
            full_res = info.get("full_res_frame") if isinstance(info, dict) else None
            if full_res is not None:
                full_frames.append(full_res)
            with torch.no_grad():
                embed = agent.encoder(obs_t)
                state = agent.rssm.observe(embed, action_onehot, state)
                feat = torch.cat([state['deter'], state['stoch'].reshape(1, -1)], dim=-1)
                action_dist = agent.actor(feat)
                action = action_dist.sample()
                action_onehot = torch.nn.functional.one_hot(action, action_dim).float()
            recon = agent.decoder(feat).sigmoid()
            pred_alive = None
            pred_xy = None
            if agent.sprite_head is not None:
                sprite_out = agent.sprite_head(feat).reshape(agent.sprite_slots, 3)
                pred_alive = torch.sigmoid(sprite_out[..., 0]).detach().cpu().numpy()
                pred_xy = torch.sigmoid(sprite_out[..., 1:3]).detach().cpu().numpy()
            action = action.cpu().numpy()[0]
            # Step Env
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            # Add to buffer
            sprite_alive = info.get("sprite_alive") if isinstance(info, dict) else None
            sprite_xy = info.get("sprite_xy") if isinstance(info, dict) else None
            buffer.add(
                obs,
                action_onehot.cpu().numpy()[0],
                reward,
                done,
                sprite_alive=sprite_alive,
                sprite_xy=sprite_xy,
            )
            recon_np = recon.squeeze(0).detach().cpu().numpy()
            if recon_np.ndim == 3 and recon_np.shape[0] in (1, 3):
                recon_np = np.transpose(recon_np, (1, 2, 0))
            if recon_np.ndim == 2:
                recon_np = recon_np[:, :, None]
            if recon_np.shape[-1] == 1:
                recon_np = np.repeat(recon_np, 3, axis=2)
            recon_np = (recon_np * 255.0).astype(np.uint8)
            recon_frames.append(recon_np)
            if pred_alive is not None and pred_xy is not None:
                sprite_frame = np.zeros_like(recon_np)
                sprite_frame = _draw_pred_sprites(sprite_frame, pred_alive, pred_xy, color=(0, 255, 0))
                combined_frame = _draw_pred_sprites(recon_np.copy(), pred_alive, pred_xy, color=(0, 255, 0))
                if sprite_alive is not None and sprite_xy is not None:
                    if isinstance(sprite_xy, np.ndarray) and sprite_xy.shape[0] > 0:
                        mario_x = info.get("x_pos") if isinstance(info, dict) else None
                        mario_y = info.get("y_pos") if isinstance(info, dict) else None
                        if mario_x is not None and mario_y is not None:
                            mario_pos = [((float(mario_x) + 8.0) / 256.0 * (recon_np.shape[1] - 1),
                                          (float(mario_y) + 16.0) / 224.0 * (recon_np.shape[0] - 1))]
                            sprite_frame = _draw_boxes(sprite_frame, mario_pos, color=(255, 0, 0), half_size=6)
                            combined_frame = _draw_boxes(combined_frame, mario_pos, color=(255, 0, 0), half_size=6)
                sprite_frames.append(sprite_frame)
                recon_combined_frames.append(combined_frame)
            if args.dump_obs_once and not dumped_obs:
                frame = next_obs
                if frame.ndim == 3 and frame.shape[0] in (1, 3):
                    frame = np.transpose(frame, (1, 2, 0))
                if frame.ndim == 2:
                    frame = frame[:, :, None]
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=2)
                cv2.imwrite("debug_obs_raw.png", frame)
                print("Wrote debug_obs_raw.png")
                dumped_obs = True
            if recon_frames and len(recon_frames) == 1:
                print(
                    "[Recon] min/max/mean:",
                    recon.min().item(),
                    recon.max().item(),
                    recon.mean().item(),
                )
            obs = next_obs
            obs_t = torch.from_numpy(obs).float().to(device) / 255.0
            step += 1
            # Train
            if step > 1000 and step % 5 == 0:
                batch = buffer.sample(16)
                metrics = agent.train_step(*[t.to(device) for t in batch])
                if step % 100 == 0:
                    print(f"Step {step} | Reward: {episode_reward:.2f} | Model Loss: {metrics['loss_model']:.4f} | Actor Loss: {metrics['loss_actor']:.4f}")
                if step % 5000 == 0:
                    os.makedirs('checkpoints', exist_ok=True)
                    torch.save(agent.state_dict(), f'checkpoints/dreamer_mario_{step}.pt')
                    print(f"Saved checkpoint at step {step}")
        # Save episode videos
        os.makedirs('videos', exist_ok=True)
        video_path = os.path.join('videos', f'episode_{episode_idx:05d}.mp4')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        for f in frames:
            out.write(f)
        out.release()
        if full_frames:
            full_video_path = os.path.join('videos', f'episode_{episode_idx:05d}_full.mp4')
            full_height, full_width = full_frames[0].shape[:2]
            out_full = cv2.VideoWriter(full_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (full_width, full_height))
            for f in full_frames:
                out_full.write(f)
            out_full.release()
        if recon_frames:
            recon_video_path = os.path.join('videos', f'episode_{episode_idx:05d}_recon.mp4')
            recon_height, recon_width = recon_frames[0].shape[:2]
            out_recon = cv2.VideoWriter(recon_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (recon_width, recon_height))
            for f in recon_frames:
                out_recon.write(f)
            out_recon.release()
        if sprite_frames:
            sprite_video_path = os.path.join('videos', f'episode_{episode_idx:05d}_recon_sprites.mp4')
            sprite_height, sprite_width = sprite_frames[0].shape[:2]
            out_sprite = cv2.VideoWriter(sprite_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (sprite_width, sprite_height))
            for f in sprite_frames:
                out_sprite.write(f)
            out_sprite.release()
        if recon_combined_frames:
            combined_video_path = os.path.join('videos', f'episode_{episode_idx:05d}_recon_combined.mp4')
            combined_height, combined_width = recon_combined_frames[0].shape[:2]
            out_combined = cv2.VideoWriter(combined_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (combined_width, combined_height))
            for f in recon_combined_frames:
                out_combined.write(f)
            out_combined.release()
        print(f"Saved episode video: {video_path}")
        if full_frames:
            print(f"Saved full-res episode video: {full_video_path}")
        if recon_frames:
            print(f"Saved recon episode video: {recon_video_path}")
        if sprite_frames:
            print(f"Saved sprite recon episode video: {sprite_video_path}")
        if recon_combined_frames:
            print(f"Saved combined recon episode video: {combined_video_path}")
        print(f"Episode Finished | Total Reward: {episode_reward:.2f}")
        episode_idx += 1

if __name__ == "__main__":
    main()
