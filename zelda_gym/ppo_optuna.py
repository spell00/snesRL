import optuna
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from mario_bizhawk_env import MarioBizHawkEnv
from stable_baselines3.common.callbacks import BaseCallback

class BestModelSaveCallback(BaseCallback):
    def __init__(self, save_path, score_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.score_path = score_path
        self.best_mean_reward = -float('inf')
        self.episode_rewards = []

    def _on_step(self) -> bool:
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

class EarlyStopOnRewardDrop(BaseCallback):
    def __init__(self, patience=100, verbose=0):
        super().__init__(verbose)
        self.patience = patience
        self._last_rew = None
        self._drop_count = 0
    def _on_step(self) -> bool:
        # Track episode rewards from infos
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                rew = info['episode']['r']
                if self._last_rew is not None and rew < self._last_rew:
                    self._drop_count += 1
                else:
                    self._drop_count = 0
                self._last_rew = rew
                if self._drop_count >= self.patience:
                    print(f"Early stopping: reward dropped {self.patience} times in a row.")
                    return False
        return True

import argparse

def make_env(rank, args):
    def _init():
        base_env = MarioBizHawkEnv(
            rank=rank,
            frameskip=args['frameskip'],
            obs_size=args['obs_size'],
            verbose=args['verbose'],
            model_name="ppo_optuna",
            action_type=args['action_type'],
            cell_bonus_mode=args['cell_bonus_mode'],
            enable_cell_exploration=args['enable_cell_exploration'],
            novelty_enabled=args['novelty_enabled'],
            reset_mode=args.get('reset_mode', None)
        )
        # Level selection logic
        orig_reset = base_env.reset
        def reset_with_level(*, seed=None, options=None):
            if args.get('level', 0) == 0:
                return orig_reset(seed=seed, options=options)
            else:
                savestate = base_env.get_savestate_by_index(args['level'])
                opts = options.copy() if options else {}
                opts["savestate"] = savestate
                return orig_reset(seed=seed, options=opts)
        base_env.reset = reset_with_level
        env = Monitor(base_env, filename=os.path.join("./logs/ppo_optuna", f"ppo_optuna_{rank}"))
        return env
    return _init

def objective(trial):
    # Suggest hyperparameters
    # Use global args for env/reward params
    global args
    hparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [64, 128, 256, 512]),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'n_epochs': trial.suggest_categorical('n_epochs', [4, 8, 16]),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.98),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
        'ent_coef': trial.suggest_float('ent_coef', 0.01, 0.5, log=True),
        'policy': args.policy,
        'frameskip': args.frameskip,
        'obs_size': args.obs_size,
        'verbose': args.verbose,
        'action_type': args.action_type,
        'cell_bonus_mode': args.cell_bonus_mode,
        'enable_cell_exploration': args.enable_cell_exploration,
        'novelty_enabled': args.novelty_enabled,
        'n_envs': args.n_envs,
        'timesteps': args.timesteps
    }
    env_fns = [make_env(i, hparams) for i in range(hparams['n_envs'])]
    if hparams['n_envs'] > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        hparams['policy'], env,
        learning_rate=hparams['learning_rate'],
        n_steps=hparams['n_steps'],
        batch_size=hparams['batch_size'],
        n_epochs=hparams['n_epochs'],
        gamma=hparams['gamma'],
        gae_lambda=hparams['gae_lambda'],
        clip_range=hparams['clip_range'],
        ent_coef=hparams['ent_coef'],
        verbose=0,
        tensorboard_log="./logs/ppo_optuna",
        device=device
    )
    # Setup best model callback for this trial
    trial_dir = os.path.join("./models/ppo_optuna", f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    best_model_path = os.path.join(trial_dir, "best_model")
    best_score_path = os.path.join(trial_dir, "best_score.txt")
    best_callback = BestModelSaveCallback(best_model_path, best_score_path, verbose=1)
    early_stop_callback = EarlyStopOnRewardDrop(patience=100, verbose=1)
    # Compose callbacks: run both
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList([best_callback, early_stop_callback])
    model.learn(total_timesteps=hparams['timesteps'], callback=callback)
    # Evaluate: use last episode reward seen by early_stop_callback
    ep_rew_mean = getattr(early_stop_callback, '_last_rew', -float('inf'))
    env.close()
    return ep_rew_mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--timesteps", type=int, default=20000, help="Timesteps per trial")
    parser.add_argument("--level", type=int, default=0, help="Level index: 0=random, 1=first, 2=second, ...")
    parser.add_argument("--n_envs", type=int, default=10, help="Number of parallel environments")
    parser.add_argument("--frameskip", type=int, default=6, help="Frameskip value")
    parser.add_argument("--obs_size", type=int, default=64, help="Observation size (pixels)")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
    parser.add_argument("--action_type", type=str, default="box", choices=["box", "discrete", "multibinary"], help="Action type")
    parser.add_argument("--cell_bonus_mode", type=str, default="linear", help="Cell bonus mode")
    parser.add_argument("--enable_cell_exploration", action="store_true", help="Enable cell exploration bonus")
    parser.add_argument("--novelty_enabled", action="store_true", help="Enable novelty bonus")
    parser.add_argument("--policy", type=str, default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"], help="Policy architecture (CnnPolicy or MlpPolicy)")
    parser.add_argument("--reset_mode", type=str, default=None, choices=[None, "soft", "hard"], help="Reset mode for the emulator/game state on each episode reset (soft/hard)")
    args, unknown = parser.parse_known_args()

    def objective_with_level(trial):
        global make_env
        def inject_level(hparams):
            hparams = dict(hparams)
            hparams['level'] = args.level
            hparams['timesteps'] = args.timesteps
            return hparams
        # Patch make_env to use level
        orig_make_env = make_env
        def make_env_with_level(rank, hparams):
            return orig_make_env(rank, inject_level(hparams))
        make_env = make_env_with_level
        return objective(trial)

    from tqdm import tqdm
    study = optuna.create_study(direction="maximize")
    with tqdm(total=args.n_trials, desc="Optuna Trials") as pbar:
        def progress_callback(study, trial):
            pbar.update(1)
            # Print best mean reward (rollback score) for this trial if available
            if hasattr(trial, 'user_attrs') and 'best_mean_reward' in trial.user_attrs:
                print(f"Trial {trial.number} rollback score: {trial.user_attrs['best_mean_reward']}")
            elif hasattr(study, 'best_value'):
                print(f"Trial {trial.number} best value so far: {study.best_value}")
        study.optimize(objective_with_level, n_trials=args.n_trials, callbacks=[progress_callback])
    print("Best trial:", study.best_trial.params)
