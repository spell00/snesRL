import torch
import numpy as np
import multiprocessing as mp
from agent import DreamerAgent
from buffer import ReplayBuffer

def env_worker(rank, env_fn, pipe, obs_shape, action_dim, device):
    env = env_fn(rank)
    
    while True:
        try:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                obs, info = env.reset()
                pipe.send((obs, info))
            elif cmd == 'step':
                action = data
                next_obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    next_obs, _ = env.reset()
                pipe.send((next_obs, reward, terminated, truncated, info))
            elif cmd == 'close':
                env.close()
                break
        except Exception as e:
            print(f"Worker {rank} error: {e}")
            try:
                env.close()
            except:
                pass
            print(f"Worker {rank} recreating environment after error...")
            try:
                env = env_fn(rank)
                obs, info = env.reset()
                if cmd == 'step':
                    pipe.send((obs, 0.0, False, False, {"error": str(e), "recreated": True}))
                else:
                    pipe.send((obs, info))
            except Exception as e2:
                print(f"Worker {rank} critical failure during recreation: {e2}")
                # If we really can't recover, we're in trouble. 
                # But at least don't hang the parent if we can help it.
                break

class ParallelEnvs:
    def __init__(self, n_envs, env_fn, obs_shape, action_dim, device):
        self.n_envs = n_envs
        self.pipes, worker_pipes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.processes = [
            mp.Process(target=env_worker, args=(i, env_fn, worker_pipes[i], obs_shape, action_dim, device))
            for i in range(n_envs)
        ]
        for p in self.processes:
            p.start()

    def reset(self):
        for pipe in self.pipes:
            pipe.send(('reset', None))
        results = [pipe.recv() for pipe in self.pipes]
        obs_list, info_list = zip(*results)
        return np.stack(obs_list), info_list

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', action))
        results = [pipe.recv() for pipe in self.pipes]
        obs, rewards, terms, truncs, infos = zip(*results)
        return np.stack(obs), np.array(rewards), np.array(terms), np.array(truncs), infos

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except:
                pass
        for p in self.processes:
            p.join()
