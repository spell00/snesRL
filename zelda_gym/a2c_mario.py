import os
import argparse
import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from mario_bizhawk_env import MarioBizHawkEnv

LOG_DIR = "./logs/a2c"
MODEL_DIR = "./models/a2c"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel training environments")
    parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--eval_freq", type=int, default=10_000, help="Evaluate every N steps")
    parser.add_argument("--n_eval_episodes", type=int, default=3, help="Episodes per evaluation")

    parser.add_argument("--screenshot_every", type=int, default=2, help="Screenshot every N actions (debug only)")
    parser.add_argument("--obs_size", type=int, default=64, help="Resize observations to (size x size)")
    parser.add_argument("--no_load", action="store_true", help="Start fresh (do not resume checkpoint)")
    parser.add_argument("--frameskip", type=int, default=2, help="Frameskip for MarioBizHawkEnv")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity for A2C and environment")

    parser.add_argument("--reset_mode", type=str, default="soft", choices=["soft", "hard"],
                        help="Training reset mode")
    parser.add_argument("--action_type", type=str, default="discrete",
                        choices=["discrete", "multibinary", "box"], help="Action space type")

    parser.add_argument("--level", type=int, default=3, help="Level index: 0=random, 1=first, 2=second, ...")

    parser.add_argument("--enable_cell_exploration", action="store_true", help="Enable cell-based exploration bonus")
    parser.add_argument("--novelty_enabled", action="store_true", help="Enable cheap novelty bonus (default False here)")
    parser.add_argument("--exploration_bonus", type=float, default=10.0, help="Exploration bonus amount")
    parser.add_argument("--cell_bonus_mode", type=str, default="linear",
                        choices=["linear", "exp", "log1p", "constant", "other"], help="Cell bonus mode")

    parser.add_argument("--death_penalty", type=float, default=-50.0, help="Penalty for dying")
    parser.add_argument("--win_bonus", type=float, default=500.0, help="Bonus for winning")
    parser.add_argument("--progress_per_pixel", type=float, default=0.0, help="Reward per pixel progress to the right")

    parser.add_argument("--debug_screenshots", action="store_true", help="Save screenshots frequently (slow)")
    parser.add_argument("--use_novelty", action="store_true", help="Use RewardSMW novelty inside env.rewarder")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for A2C")

    return parser.parse_args()


def latest_checkpoint_in(dir_path: str):
    ckpts = [f for f in os.listdir(dir_path) if f.startswith("a2c_") and f.endswith("_steps.zip")]
    if not ckpts:
        return None
    try:
        ckpts.sort(key=lambda x: int(x.split("_")[1]))
        return os.path.join(dir_path, ckpts[-1])
    except Exception:
        return None


def wrap_env(env):
    # SB3 CNN expects (C,H,W) in Vec envs; we have (H,W,C)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


def make_train_env_factory(args, rank: int):
    def _init():
        base = MarioBizHawkEnv(
            rank=rank,
            headless=False,
            frameskip=args.frameskip,
            screenshot_every=(args.screenshot_every if args.debug_screenshots else 1_000_000),
            obs_size=args.obs_size,
            verbose=args.verbose,
            reset_mode=args.reset_mode,
            model_name="a2c_train",
            action_type=args.action_type,
            enable_cell_exploration=args.enable_cell_exploration,
            novelty_enabled=args.novelty_enabled,
            exploration_bonus=args.exploration_bonus,
            cell_bonus_mode=args.cell_bonus_mode,
            death_penalty=args.death_penalty,
            win_bonus=args.win_bonus,
            progress_per_pixel=args.progress_per_pixel,
        )

        # apply RewardSMW novelty flag if you want
        base.rewarder.use_novelty = bool(args.use_novelty)

        # force fixed level selection (or random if level==0) on every reset
        orig_reset = base.reset

        def reset_with_level(*, seed=None, options=None):
            opts = dict(options) if options else {}
            if args.level != 0:
                opts["savestate"] = base.get_savestate_by_index(args.level)
            return orig_reset(seed=seed, options=opts)

        base.reset = reset_with_level

        return Monitor(base, filename=os.path.join(LOG_DIR, f"train_rank{rank}_hz{args.frameskip}"))

    return _init


def make_eval_env_factory(args, eval_rank: int):
    def _init():
        # IMPORTANT: eval env should be isolated + hard reset to avoid wedge
        base = MarioBizHawkEnv(
            rank=eval_rank,
            headless=False,
            frameskip=args.frameskip,
            screenshot_every=1_000_000,
            obs_size=args.obs_size,
            verbose=0,
            reset_mode="hard",          # force hard for eval
            model_name="a2c_eval",
            action_type=args.action_type,
            enable_cell_exploration=args.enable_cell_exploration,
            novelty_enabled=args.novelty_enabled,
            exploration_bonus=args.exploration_bonus,
            cell_bonus_mode=args.cell_bonus_mode,
            death_penalty=args.death_penalty,
            win_bonus=args.win_bonus,
            progress_per_pixel=args.progress_per_pixel,
        )

        base.rewarder.use_novelty = bool(args.use_novelty)

        orig_reset = base.reset

        def reset_with_level(*, seed=None, options=None):
            opts = dict(options) if options else {}
            if args.level != 0:
                opts["savestate"] = base.get_savestate_by_index(args.level)
            return orig_reset(seed=seed, options=opts)

        base.reset = reset_with_level

        return Monitor(base, filename=os.path.join(LOG_DIR, f"eval_rank{eval_rank}_hz{args.frameskip}"))

    return _init


class HardRecreateEvalCallback(EvalCallback):
    """
    Recreate eval env right before evaluation to avoid stuck BizHawk/Lua.
    """
    def __init__(self, make_eval_vec_env_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._make_eval_vec_env_fn = make_eval_vec_env_fn

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq) == 0:
            try:
                self.eval_env.close()
            except Exception:
                pass
            self.eval_env = self._make_eval_vec_env_fn()
        return super()._on_step()


def train():
    args = parse_args()

    hz_dir = os.path.join(MODEL_DIR, f"hz{args.frameskip}")
    os.makedirs(hz_dir, exist_ok=True)

    # ---- training vec env ----
    train_fns = [make_train_env_factory(args, rank=i) for i in range(args.n_envs)]
    if args.n_envs > 1:
        train_env = SubprocVecEnv(train_fns)
    else:
        train_env = DummyVecEnv(train_fns)

    train_env = wrap_env(train_env)

    # ---- model ----
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = None if args.no_load else latest_checkpoint_in(hz_dir)
    if ckpt:
        print(f"RESUMING TRAINING from checkpoint: {ckpt}")
        model = A2C.load(ckpt, env=train_env, device=device)
    else:
        print("Starting FRESH training...")
        model = A2C(
            "CnnPolicy",
            train_env,
            learning_rate=1e-4,
            ent_coef=args.ent_coef,
            verbose=args.verbose,
            tensorboard_log=LOG_DIR,
            device=device,
        )

    # ---- eval vec env factory ----
    # Use a rank that will NEVER overlap training ranks
    eval_rank = 999

    def make_eval_vec_env():
        e = DummyVecEnv([make_eval_env_factory(args, eval_rank=eval_rank)])
        e = wrap_env(e)
        return e

    eval_env = make_eval_vec_env()

    eval_callback = HardRecreateEvalCallback(
        make_eval_vec_env_fn=make_eval_vec_env,
        eval_env=eval_env,
        best_model_save_path=hz_dir,
        log_path=LOG_DIR,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    print(f"Starting training for {args.timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    finally:
        # save final
        model.save(os.path.join(hz_dir, "a2c_final"))

        # close envs
        try:
            train_env.close()
        except Exception as e:
            print(f"Warning: error closing training env: {e}")

        try:
            eval_callback.eval_env.close()
        except Exception:
            pass

        # last-resort cleanup (Windows): kill any remaining BizHawk
        try:
            import subprocess
            subprocess.call(
                ["taskkill", "/F", "/IM", "EmuHawk.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Warning: could not force-kill BizHawk: {e}")


if __name__ == "__main__":
    train()
