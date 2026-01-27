import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import os
import argparse
from mario_bizhawk_ram_env import MarioBizHawkRamEnv

LOG_DIR = "./logs/ppo_ram"
MODEL_DIR = "./models/ppo_ram"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--frameskip", type=int, default=6, help="Frameskip (action repeat)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--no_load", "--no-load", action="store_true", help="Start fresh without loading the latest checkpoint")
    args = parser.parse_args()

    HZ_MODEL_DIR = os.path.join(MODEL_DIR, f"hz{args.frameskip}")
    os.makedirs(HZ_MODEL_DIR, exist_ok=True)

    def make_env_factory(rank, headless=False):
        def _init():
            env = MarioBizHawkRamEnv(
                rank=rank,
                headless=headless,
                frameskip=args.frameskip,
                verbose=args.verbose
            )
            env = Monitor(env, filename=os.path.join(LOG_DIR, f"{rank}_hz{args.frameskip}"))
            return env
        return _init

    env_fns = [make_env_factory(i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    checkpoints = [f for f in os.listdir(HZ_MODEL_DIR) if f.startswith("ppo_mario_ram_") and f.endswith("_steps.zip")]
    latest_checkpoint = None
    if checkpoints and not args.no_load:
        try:
            checkpoints.sort(key=lambda x: int(x.split('_')[4]))
            latest_checkpoint = os.path.join(HZ_MODEL_DIR, checkpoints[-1])
        except (ValueError, IndexError): pass

    if latest_checkpoint:
        print(f"RESUMING TRAINING from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, device=device)
    else:
        print(f"Starting FRESH RAM-based training...")
        model = PPO(
            "MlpPolicy",  # Use MLP for RAM
            env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=args.verbose,
            tensorboard_log=LOG_DIR,
            device=device
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000 // args.n_envs,
        save_path=HZ_MODEL_DIR,
        name_prefix="ppo_mario_ram"
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            log_interval=1,
            reset_num_timesteps=(latest_checkpoint is None)
        )
    finally:
        model.save(f"{HZ_MODEL_DIR}/ppo_mario_ram_final")
        env.close()

if __name__ == "__main__":
    train()
