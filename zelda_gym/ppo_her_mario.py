import argparse
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from mario_bizhawk_env import MarioBizHawkEnv

# WARNING: HER is not officially supported for PPO in Stable Baselines3. This script is experimental and may not work as intended.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--policy", type=str, default="cnn", choices=["cnn", "mlp"], help="Policy type: 'cnn' for CnnPolicy, 'mlp' for MlpPolicy")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for exploration (PPO ent_coef)")
    parser.add_argument("--frameskip", type=int, default=6)
    parser.add_argument("--obs_size", type=int, default=64)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="box", choices=["box"])
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--cell_bonus_mode", type=str, default="linear", choices=["linear", "exp", "log1p"])
    parser.add_argument("--enable_cell_exploration", action="store_true", help="Enable cell-based exploration bonus")
    parser.add_argument("--novelty_enabled", action="store_true", help="Enable SSIM-based novelty bonus (default True)")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--no_load", "--no-load", action="store_true", help="Start fresh (do not load checkpoint)")
    args = parser.parse_args()

    LOG_DIR = "./logs/ppo_her"
    MODEL_DIR = "./models/ppo_her"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    HZ_MODEL_DIR = os.path.join(MODEL_DIR, f"hz{args.frameskip}")
    os.makedirs(HZ_MODEL_DIR, exist_ok=True)

    def make_env_factory(rank):
        def _init():
            base_env = MarioBizHawkEnv(
                rank=rank,
                frameskip=args.frameskip,
                obs_size=args.obs_size,
                verbose=args.verbose,
                model_name="ppo_her",
                action_type=args.action_type,
                cell_bonus_mode=args.cell_bonus_mode,
                enable_cell_exploration=args.enable_cell_exploration,
                novelty_enabled=args.novelty_enabled
            )
            orig_reset = base_env.reset
            def reset_with_level(*, seed=None, options=None):
                if args.level == 0:
                    return orig_reset(seed=seed, options=options)
                else:
                    savestate = base_env.get_savestate_by_index(args.level)
                    opts = options.copy() if options else {}
                    opts["savestate"] = savestate
                    return orig_reset(seed=seed, options=opts)
            base_env.reset = reset_with_level
            env = Monitor(base_env, filename=os.path.join(LOG_DIR, f"ppo_her_{rank}"))
            return env
        return _init

    env_fns = [make_env_factory(i) for i in range(args.n_envs)]
    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # HER is not supported for PPO in Stable Baselines3. This will run PPO as normal.
    policy_class = "CnnPolicy" if args.policy == "cnn" else "MlpPolicy"
    ckpt_prefix = f"ppo_her_{args.policy}_mario_steps"
    ckpt_path = os.path.join(HZ_MODEL_DIR, f"{ckpt_prefix}.zip")
    latest_checkpoint = ckpt_path if os.path.exists(ckpt_path) and not getattr(args, 'no_load', False) else None
    checkpoints = [f for f in os.listdir(HZ_MODEL_DIR) if f.startswith(ckpt_prefix) and f.endswith("_steps.zip")]
    latest_checkpoint = None
    if checkpoints and not args.no_load:
        try:
            checkpoints.sort(key=lambda x: int(x.split('_')[5]))
            latest_checkpoint = os.path.join(HZ_MODEL_DIR, checkpoints[-1])
        except (ValueError, IndexError):
            pass

    if latest_checkpoint:
        print(f"RESUMING TRAINING from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, device=device)
    else:
        print(f"Starting FRESH training...")
        model = PPO(
            policy_class, env,
            verbose=args.verbose,
            # learning_rate=3e-4,
            # n_steps=128,
            # batch_size=32,
            # n_epochs=4,
            # gamma=0.99,
            # gae_lambda=0.95,
            # clip_range=0.2,
            # ent_coef=args.ent_coef,
            tensorboard_log=LOG_DIR,
            device=device
        )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // args.n_envs,
        save_path=HZ_MODEL_DIR,
        name_prefix=f"ppo_her_{args.policy}_mario_steps"
    )
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    model.save(ckpt_path)
    env.close()

if __name__ == "__main__":
    main()
