import argparse
import multiprocessing as mp
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from mario_bizhawk_env import MarioBizHawkEnv
import time

def make_eval_env(args):
    def _init():
        env = MarioBizHawkEnv(
            rank=args.eval_rank,
            headless=False,
            frameskip=args.frameskip,
            screenshot_every=1000000,
            obs_size=args.obs_size,
            verbose=0,
            reset_mode='hard',
            model_name='a2c_eval',
            action_type=args.action_type,
            enable_cell_exploration=args.enable_cell_exploration,
            novelty_enabled=args.novelty_enabled,
            exploration_bonus=args.exploration_bonus,
            cell_bonus_mode=args.cell_bonus_mode,
            death_penalty=args.death_penalty,
            win_bonus=args.win_bonus,
            progress_per_pixel=args.progress_per_pixel,
        )
        return Monitor(env)
    return _init

def eval_worker(args):
    eval_env = DummyVecEnv([make_eval_env(args)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)
    model = A2C.load(args.model_path, env=eval_env)
    episode_rewards = []
    for ep in range(args.n_eval_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
        print(f"[EVAL] Episode {ep+1}: reward={total_reward}")
        episode_rewards.append(total_reward)
    print(f"[EVAL] Mean reward over {args.n_eval_episodes} episodes: {sum(episode_rewards)/len(episode_rewards)}")
    eval_env.close()

def start_eval_in_parallel(args):
    p = mp.Process(target=eval_worker, args=(args,))
    p.start()
    return p

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--eval_rank', type=int, default=99)
    parser.add_argument('--n_eval_episodes', type=int, default=5)
    # Add other args as needed (frameskip, obs_size, etc.)
    args = parser.parse_args()
    start_eval_in_parallel(args)
    print("[MAIN] Evaluation started in parallel process.")
    # Optionally, do other work here or join the process
    # p.join()  # Uncomment to wait for eval to finish
