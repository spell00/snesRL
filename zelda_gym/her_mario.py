import argparse
import os
import torch
from stable_baselines3 import HER, DDPG
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from mario_bizhawk_env import MarioBizHawkEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--policy", type=str, default="cnn", choices=["cnn", "mlp"], help="Policy type: 'cnn' for CnnPolicy, 'mlp' for MlpPolicy")
    parser.add_argument("--frameskip", type=int, default=6)
    parser.add_argument("--obs_size", type=int, default=64)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="box", choices=["box"])
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--cell_bonus_mode", type=str, default="linear", choices=["linear", "exp", "log1p"])
    parser.add_argument("--enable_cell_exploration", action="store_true", help="Enable cell-based exploration bonus")
    parser.add_argument("--novelty_enabled", action="store_true", help="Enable SSIM-based novelty bonus (default True)")
    parser.add_argument("--n_envs", type=int, default=10, help="Number of parallel environments")
    args = parser.parse_args()

    LOG_DIR = "./logs/her"
    MODEL_DIR = "./models/her"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    def make_env_factory(rank):
        def _init():
            base_env = MarioBizHawkEnv(
                rank=rank,
                frameskip=args.frameskip,
                obs_size=args.obs_size,
                verbose=args.verbose,
                model_name="her",
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
            env = Monitor(base_env, filename=os.path.join(LOG_DIR, f"her_{rank}"))
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_class = "CnnPolicy" if args.policy == "cnn" else "MlpPolicy"
    model = HER(
        policy_class, env,
        DDPG,
        n_sampled_goal=4,
        goal_selection_strategy=GoalSelectionStrategy.FUTURE,
        verbose=args.verbose,
        tensorboard_log=LOG_DIR,
        device=device
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(MODEL_DIR, f"her_{args.policy}_mario_final"))
    env.close()

if __name__ == "__main__":
    main()
