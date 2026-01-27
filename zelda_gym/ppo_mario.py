
import gymnasium as gym
from stable_baselines3 import PPO
try:
    from sb3_contrib import RecurrentPPO
    PPORecurrent = RecurrentPPO
except ImportError:
    PPORecurrent = None
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
import os
import argparse
from mario_bizhawk_env import MarioBizHawkEnv

# Create directories for logs and models
LOG_DIR = "./logs/ppo"
MODEL_DIR = "./models/ppo"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--frameskip", "--fps", type=int, default=6, help="Agent actions per second")
    parser.add_argument("--screenshot_every", type=int, default=2, help="Screenshot every N actions")
    parser.add_argument("--obs_size", type=int, default=64, help="Resize observations to (size x size)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--no_load", "--no-load", action="store_true", help="Start fresh without loading the latest checkpoint")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for exploration (PPO ent_coef)")
    parser.add_argument("--reset_mode", type=str, default="soft", choices=["soft", "hard"], help="Reset mode: 'soft' (Lua RESET) or 'hard' (restart BizHawk)")
    parser.add_argument("--policy", type=str, default="cnn", choices=["cnn", "mlp", "recurrent"], help="Policy type: 'cnn', 'mlp', or 'recurrent' (LSTM)")
    args = parser.parse_args()

    HZ_MODEL_DIR = os.path.join(MODEL_DIR, f"hz{args.frameskip}")
    os.makedirs(HZ_MODEL_DIR, exist_ok=True)

    def make_env_factory(rank, headless=False):
        def _init():
            env = MarioBizHawkEnv(
                rank=rank, 
                headless=headless, 
                frameskip=args.frameskip,
                screenshot_every=args.screenshot_every,
                obs_size=args.obs_size,
                verbose=args.verbose,
                reset_mode=args.reset_mode
            )
            env = Monitor(env, filename=os.path.join(LOG_DIR, f"{rank}_hz{args.frameskip}")) 
            return env
        return _init

    env_fns = [make_env_factory(i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    # Frame Stack remains essential for velocity
    env = VecFrameStack(env, n_stack=4)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Find latest checkpoint for the selected policy
    ckpt_prefix = f"ppo_mario_{args.policy}_"
    checkpoints = [f for f in os.listdir(HZ_MODEL_DIR) if f.startswith(ckpt_prefix) and f.endswith("_steps.zip")]
    latest_checkpoint = None
    if checkpoints and not args.no_load:
        try:
            checkpoints.sort(key=lambda x: int(x.split('_')[4]))
            latest_checkpoint = os.path.join(HZ_MODEL_DIR, checkpoints[-1])
        except (ValueError, IndexError):
            pass

    if args.policy == "recurrent":
        if PPORecurrent is None:
            raise ImportError("sb3_contrib is required for RecurrentPPO. Install with 'pip install sb3-contrib'.")
        policy_class = "RecurrentPPOPolicy"
        PPOClass = PPORecurrent
    else:
        policy_class = "CnnPolicy" if args.policy == "cnn" else "MlpPolicy"
        PPOClass = PPO

    if latest_checkpoint:
        print(f"RESUMING TRAINING from checkpoint: {latest_checkpoint}")
        model = PPOClass.load(latest_checkpoint, env=env, device=device)
    else:
        print(f"Starting FRESH training (Optimized for CPU)...")
        model = PPOClass(
            policy_class,
            env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device=device
        )


    class BestModelSaveCallback(BaseCallback):
        def __init__(self, save_path, score_path, verbose=0):
            super().__init__(verbose)
            self.save_path = save_path
            self.score_path = score_path
            self.best_mean_reward = -float('inf')
            self.episode_rewards = []

        def _on_step(self) -> bool:
            # Gather episode rewards from infos
            infos = self.locals.get('infos', [])
            for info in infos:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
            # Only check/save every 1000 steps for efficiency
            if self.n_calls % 1000 == 0 and len(self.episode_rewards) >= 100:
                mean_reward = float(sum(self.episode_rewards[-100:]) / 100)
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
                    with open(self.score_path, 'w') as f:
                        f.write(str(self.best_mean_reward))
                    if self.verbose:
                        print(f"\nNew best mean reward: {mean_reward:.2f} - model saved.")
            return True

    best_model_path = os.path.join(HZ_MODEL_DIR, f"ppo_mario_{args.policy}_best_model")
    best_score_path = os.path.join(HZ_MODEL_DIR, f"ppo_mario_{args.policy}_best_score.txt")
    best_callback = BestModelSaveCallback(best_model_path, best_score_path, verbose=1)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=best_callback,
            log_interval=1,
            reset_num_timesteps=(latest_checkpoint is None)
        )
    finally:
        model.save(f"{HZ_MODEL_DIR}/ppo_mario_{args.policy}_final")
        env.close()

if __name__ == "__main__":
    train()
