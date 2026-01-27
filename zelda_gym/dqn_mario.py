import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
import os
import argparse
from mario_bizhawk_env import MarioBizHawkEnv
import torch

# Create directories for logs and models
LOG_DIR = "./logs/dqn"
MODEL_DIR = "./models/dqn"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=100000000, help="Total training timesteps")
    # Removed action_hz, replaced with frameskip
    parser.add_argument("--screenshot_every", type=int, default=2, help="Screenshot every N actions")
    parser.add_argument("--obs_size", type=int, default=64, help="Resize observations to (size x size)")
    parser.add_argument("--no_load", "--no-load", action="store_true", help="Start fresh")
    parser.add_argument("--frameskip", type=int, default=2, help="Frameskip for MarioBizHawkEnv")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity for DQN and environment")
    parser.add_argument("--reset_mode", type=str, default="soft", choices=["soft", "hard"], help="Reset mode: 'soft' (Lua RESET) or 'hard' (restart BizHawk)")
    parser.add_argument("--action_type", type=str, default="discrete", choices=["discrete"], help="Action space type: 'discrete'")
    parser.add_argument("--level", type=int, default=3, help="Level index: 0=random, 1=first, 2=second, ...")
    parser.add_argument("--enable_cell_exploration", action="store_true", help="Enable cell-based exploration bonus")
    parser.add_argument("--novelty_enabled", action="store_true", help="Enable SSIM-based novelty bonus (default True)")
    parser.add_argument("--exploration_bonus", type=float, default=10.0, help="Cell-based exploration bonus amount")
    parser.add_argument("--cell_bonus_mode", type=str, default="linear", choices=["linear", "exp", "log1p", "constant", "other"], help="Cell bonus mode: 'linear', 'exp', or 'log1p'")
    parser.add_argument("--death_penalty", type=float, default=-50.0, help="Penalty for dying (falling in pit)")
    parser.add_argument("--win_bonus", type=float, default=500.0, help="Bonus for winning a level")
    parser.add_argument("--progress_per_pixel", type=float, default=0.0, help="Reward per pixel progress to the right")
    parser.add_argument("--use_progress_savestate", action="store_true", help="Enable progress-based savestate saving/loading")
    parser.add_argument("--debug_screenshots", action="store_true", help="Save PNG screenshots for debugging (slow)")
    parser.add_argument("--use_novelty", action="store_true", help="Use novelty bonus based on SSIM")
    args = parser.parse_args()

    # Partition models by frameskip (like a3c)
    HZ_MODEL_DIR = os.path.join(MODEL_DIR, f"hz{args.frameskip}")
    os.makedirs(HZ_MODEL_DIR, exist_ok=True)


    def make_env_factory(rank, headless=False):
        def _init():
            base_env = MarioBizHawkEnv(
                rank=rank,
                headless=headless,
                frameskip=args.frameskip,
                screenshot_every=1000000 if not args.debug_screenshots else args.screenshot_every,
                obs_size=args.obs_size,
                verbose=args.verbose,
                reset_mode=args.reset_mode,
                model_name="dqn",
                action_type=args.action_type,
                enable_cell_exploration=args.enable_cell_exploration,
                novelty_enabled=args.novelty_enabled,
                exploration_bonus=args.exploration_bonus,
                cell_bonus_mode=args.cell_bonus_mode,
                death_penalty=args.death_penalty,
                win_bonus=args.win_bonus,
                progress_per_pixel=args.progress_per_pixel
            )
            # Set use_novelty on rewarder if present in args
            if hasattr(args, 'use_novelty'):
                base_env.rewarder.use_novelty = args.use_novelty

            # Always load the original level savestate; progress savestate logic disabled
            orig_reset = base_env.reset
            def reset_with_level(*, seed=None, options=None):
                opts = options.copy() if options else {}
                if not hasattr(base_env, '_has_reset_once'):
                    base_env._has_reset_once = True
                    if args.level == 0:
                        return orig_reset(seed=seed, options=options)
                    else:
                        savestate = base_env.get_savestate_by_index(args.level)
                        opts["savestate"] = savestate
                        return orig_reset(seed=seed, options=opts)
                # After first reset, always load the original level savestate (no progress savestate)
                savestate = base_env.get_savestate_by_index(args.level) if args.level != 0 else None
                if savestate is not None:
                    opts["savestate"] = savestate
                return orig_reset(seed=seed, options=opts)
            base_env.reset = reset_with_level
            env = Monitor(base_env, filename=os.path.join(LOG_DIR, f"{rank}_hz{args.frameskip}"))
            return env
        return _init

    env_fns = [make_env_factory(i) for i in range(args.n_envs)]
    if args.n_envs >= 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints = [f for f in os.listdir(HZ_MODEL_DIR) if f.startswith("dqn_") and f.endswith("_steps.zip")]
    latest_checkpoint = None
    if checkpoints and not args.no_load:
        try:
            checkpoints.sort(key=lambda x: int(x.split('_')[1]))
            latest_checkpoint = os.path.join(HZ_MODEL_DIR, checkpoints[-1])
        except (ValueError, IndexError):
            pass
    if latest_checkpoint:
        print(f"RESUMING TRAINING from checkpoint: {latest_checkpoint}")
        model = DQN.load(latest_checkpoint, env=env, device=device)
    else:
        print(f"Starting FRESH training...")
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            verbose=args.verbose,
            tensorboard_log=LOG_DIR,
            device=device
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=HZ_MODEL_DIR,
        name_prefix="dqn"
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    finally:
        model.save(f"{HZ_MODEL_DIR}/dqn_final")
        env.close()

if __name__ == "__main__":
    train()
