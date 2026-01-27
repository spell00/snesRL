import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
import os
import argparse
from mario_bizhawk_env import MarioBizHawkEnv

# Create directories for logs and models
LOG_DIR = "./logs/a3c"
MODEL_DIR = "./models/a3c"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=10000000, help="Total training timesteps")
    parser.add_argument("--frameskip", type=int, default=6, help="Agent actions per second (frameskip)")
    parser.add_argument("--screenshot_every", type=int, default=2, help="Screenshot every N actions")
    parser.add_argument("--obs_size", type=int, default=64, help="Resize observations to (size x size)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--no_load", "--no-load", action="store_true", help="Start fresh without loading the latest checkpoint")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for exploration (A2C ent_coef)")
    parser.add_argument("--reset_mode", type=str, default="soft", choices=["soft", "hard"], help="Reset mode: 'soft' (Lua RESET) or 'hard' (restart BizHawk)")
    parser.add_argument("--action_type", type=str, default="discrete", choices=["discrete", "multibinary"], help="Action space type: 'discrete' or 'multibinary'")
    parser.add_argument("--level", type=int, default=3, help="Level index: 0=random, 1=first, 2=second, ...")
    parser.add_argument("--cell_bonus_mode", type=str, default="linear", choices=["linear", "exp", "log1p"], help="Cell bonus mode: 'linear', 'exp', or 'log1p'")
    parser.add_argument("--enable_cell_exploration", action="store_true", help="Enable cell-based exploration bonus")
    parser.add_argument("--novelty_enabled", action="store_true", help="Enable SSIM-based novelty bonus (default True)")
    parser.add_argument("--tf_rmsprop", action="store_true", help="Use TensorFlow-like RMSprop optimizer for A2C (RMSpropTFLike)")
    args = parser.parse_args()

    HZ_MODEL_DIR = f'{MODEL_DIR}/hz{args.frameskip}'
    os.makedirs(HZ_MODEL_DIR, exist_ok=True)

    def make_env_factory(rank, headless=False):
        def _init():
            base_env = MarioBizHawkEnv(
                rank=rank,
                headless=headless,
                frameskip=args.frameskip,
                screenshot_every=args.screenshot_every,
                obs_size=args.obs_size,
                verbose=args.verbose,
                reset_mode=args.reset_mode,
                model_name="a3c",
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
            env = Monitor(base_env, filename=os.path.join(LOG_DIR, f"{rank}_hz{args.frameskip}"))
            return env
        return _init

    env_fns = [make_env_factory(i) for i in range(args.n_envs)]
    if args.n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv(env_fns)
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv(env_fns)

    env = VecFrameStack(env, n_stack=4)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    checkpoints = [f for f in os.listdir(HZ_MODEL_DIR) if f.startswith("a3c_mario_cnn_") and f.endswith("_steps.zip")]
    latest_checkpoint = None
    if checkpoints and not args.no_load:
        try:
            checkpoints.sort(key=lambda x: int(x.split('_')[3])) 
            latest_checkpoint = os.path.join(HZ_MODEL_DIR, checkpoints[-1])
        except (ValueError, IndexError): pass

    if latest_checkpoint:
        print(f"RESUMING TRAINING from checkpoint: {latest_checkpoint}")
        model = A2C.load(latest_checkpoint, env=env, device=device)
    else:
        print(f"Starting FRESH training (Optimized for CPU)...")
        # Improved hyperparameters for better exploration and less local minima
        policy_kwargs = None
        if args.tf_rmsprop:
            from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
            policy_kwargs = dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
        model = A2C(
            "CnnPolicy",
            env,
            learning_rate=3e-4,  # Lower learning rate for stability
            n_steps=128,          # Longer rollouts for better credit assignment
            gamma=0.97,           # Slightly lower gamma for more responsive updates
            gae_lambda=0.95,      # Lower lambda for less bias/variance tradeoff
            ent_coef=args.ent_coef,  # Encourage more exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            use_rms_prop=True,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device=device,
            policy_kwargs=policy_kwargs
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // args.n_envs, 
        save_path=HZ_MODEL_DIR, 
        name_prefix="a3c_mario_cnn"
    )

    from stable_baselines3.common.callbacks import BaseCallback
    class ConsoleLogCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_count = 0
        def _on_step(self) -> bool:
            for info in self.locals['infos']:
                if 'episode' in info.keys():
                    self.episode_count += 1
                    ep_rew = info['episode']['r']
                    ep_len = info['episode']['l']
                    print(f"Episode {self.episode_count} Finished! Reward: {ep_rew:.2f}, Length: {ep_len} steps.")
            return True

    # Save action_type info for reproducibility
    with open(os.path.join(HZ_MODEL_DIR, "action_type.txt"), "w") as f:
        f.write(f"{args.action_type}\n")
    print(f"Starting training for {args.timesteps} timesteps with action_type={args.action_type}...")
    try:
        model.learn(
            total_timesteps=args.timesteps, 
            callback=[checkpoint_callback, ConsoleLogCallback()], 
            log_interval=10
        )
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        model.save(f"{HZ_MODEL_DIR}/a3c_mario_final")
        print(f"Model saved to {HZ_MODEL_DIR}")
        env.close()

if __name__ == "__main__":
    train()
