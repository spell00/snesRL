import torch
import numpy as np
import multiprocessing as mp
from agent import DreamerAgent
from buffer import ReplayBuffer

def env_worker(rank, env_fn, pipe, obs_shape, action_dim, device):
    import atexit
    import signal
    import os
    import time
    env = None
    def _cleanup(*_):
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
    atexit.register(_cleanup)
    def _handle_signal(signum, frame):
        _cleanup()
        raise SystemExit(0)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    retry_interval_s = 60
    while True:
        try:
            print(f"[env_worker] Rank {rank}: Initializing environment...")
            env = env_fn(rank)
            print(f"[env_worker] Rank {rank}: Environment initialized.")
            try:
                pipe.send(("ready", {"rank": rank}))
            except Exception:
                pass
            break
        except Exception as e:
            print(f"[env_worker] Rank {rank}: Environment init failed: {e}. Retrying in {retry_interval_s}s...")
            time.sleep(retry_interval_s)
    
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
            except Exception:
                pass
            print(f"Worker {rank} recreating environment after error...")
            while True:
                try:
                    env = env_fn(rank)
                    obs, info = env.reset()
                    if cmd == 'step':
                        pipe.send((obs, 0.0, False, False, {"error": str(e), "recreated": True}))
                    else:
                        pipe.send((obs, info))
                    break
                except Exception as e2:
                    print(
                        f"Worker {rank} recreation failed: {e2}. "
                        f"Retrying in {retry_interval_s}s..."
                    )
                    time.sleep(retry_interval_s)

class ParallelEnvs:
    def __init__(self, n_envs, env_fn, obs_shape, action_dim, device):
        self.n_envs = n_envs
        self.env_fn = env_fn
        self.action_dim = action_dim
        self.device = device
        self.obs_shape = obs_shape
        # Use spawn context for safety
        self.ctx = mp.get_context("spawn")
        self.pipes, worker_pipes = zip(*[self.ctx.Pipe() for _ in range(n_envs)])
        self.pipes = list(self.pipes)
        self.worker_pipes = list(worker_pipes)
        self.processes = []
        self._awaiting = [False] * n_envs
        self._last_obs = [np.zeros(obs_shape, dtype=np.uint8) for _ in range(n_envs)]
        self._last_rewards = [0.0] * n_envs
        self._last_terms = [False] * n_envs
        self._last_truncs = [False] * n_envs
        self._last_infos = [{} for _ in range(n_envs)]
        for i in range(n_envs):
            print(f"[ParallelEnvs] Starting process for env rank {i}")
            proc = self.ctx.Process(target=env_worker, args=(i, env_fn, self.worker_pipes[i], obs_shape, action_dim, device))
            self.processes.append(proc)
            proc.start()
            print(f"[ParallelEnvs] Started process PID {proc.pid} for env rank {i}")
            self._wait_for_worker_ready(self.pipes[i], i, timeout=60)

    def _recv_if_ready(self, pipe, timeout=0.0):
        if pipe.poll(timeout):
            msg = pipe.recv()
            if isinstance(msg, (tuple, list)) and len(msg) > 0 and isinstance(msg[0], str) and msg[0] == "ready":
                return None
            return msg
        return None

    def _wait_for_worker_ready(self, pipe, rank, timeout=60):
        while True:
            if pipe.poll(timeout):
                msg = pipe.recv()
                if isinstance(msg, tuple) and msg[0] == "ready":
                    print(f"[ParallelEnvs] Worker {rank} reported ready.")
                    return
                # Ignore non-ready message and continue waiting.
                continue

            print(
                f"[ParallelEnvs] Waiting for worker {rank} to be ready "
                f"(no message in {timeout}s)."
            )

    def restart_worker(self, rank, timeout=60.0):
        proc = self.processes[rank]
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        except Exception:
            pass

        parent_pipe, child_pipe = self.ctx.Pipe()
        self.pipes[rank] = parent_pipe
        self.worker_pipes[rank] = child_pipe

        proc = self.ctx.Process(
            target=env_worker,
            args=(rank, self.env_fn, child_pipe, self.obs_shape, self.action_dim, self.device),
        )
        proc.start()
        self.processes[rank] = proc
        self._awaiting[rank] = False
        self._wait_for_worker_ready(parent_pipe, rank, timeout=timeout)

        parent_pipe.send(('reset', None))
        if not parent_pipe.poll(timeout):
            raise RuntimeError(f"Env {rank} reset after restart timed out.")
        msg = parent_pipe.recv()
        if not (isinstance(msg, (tuple, list)) and len(msg) == 2):
            raise RuntimeError(f"Env {rank} bad restart reset message: {type(msg)} {msg}")
        obs, info = msg
        self._last_obs[rank] = obs
        self._last_infos[rank] = info
        self._last_rewards[rank] = 0.0
        self._last_terms[rank] = True
        self._last_truncs[rank] = True
        return obs, info

    def reset(self, timeout=60.0):
        for i, pipe in enumerate(self.pipes):
            try:
                pipe.send(('reset', None))
                self._awaiting[i] = True
            except Exception:
                self._awaiting[i] = True

        obs_list = []
        info_list = []
        for i, pipe in enumerate(self.pipes):
            if self._awaiting[i]:
                if not pipe.poll(timeout):
                    obs, info = self.restart_worker(i, timeout=timeout)
                    self._last_infos[i] = dict(info, restarted=True)
                    self._awaiting[i] = False
                    obs_list.append(self._last_obs[i])
                    info_list.append(self._last_infos[i])
                    continue
                msg = pipe.recv()
                if isinstance(msg, (tuple, list)) and len(msg) == 2:
                    obs, info = msg
                    self._last_obs[i] = obs
                    self._last_infos[i] = info
                elif isinstance(msg, (tuple, list)) and len(msg) == 5:
                    obs, reward, terminated, truncated, info = msg
                    self._last_obs[i] = obs
                    self._last_rewards[i] = reward
                    self._last_terms[i] = terminated
                    self._last_truncs[i] = truncated
                    self._last_infos[i] = info
                else:
                    raise RuntimeError(f"Env {i} bad reset message: {type(msg)} {msg}")
                self._awaiting[i] = False
            obs_list.append(self._last_obs[i])
            info_list.append(self._last_infos[i])

        return np.stack(obs_list), info_list

    def step(self, actions, timeout=60.0):
        for i, (pipe, action) in enumerate(zip(self.pipes, actions)):
            if self._awaiting[i]:
                continue
            try:
                pipe.send(('step', action))
                self._awaiting[i] = True
            except Exception:
                self._awaiting[i] = True

        obs_list = []
        rewards = []
        terms = []
        truncs = []
        infos = []

        for i, pipe in enumerate(self.pipes):
            if self._awaiting[i]:
                if not pipe.poll(timeout):
                    obs, info = self.restart_worker(i, timeout=timeout)
                    self._last_infos[i] = dict(info, restarted=True)
                    self._last_rewards[i] = 0.0
                    self._last_terms[i] = True
                    self._last_truncs[i] = True
                    self._awaiting[i] = False
                    obs_list.append(self._last_obs[i])
                    rewards.append(self._last_rewards[i])
                    terms.append(self._last_terms[i])
                    truncs.append(self._last_truncs[i])
                    infos.append(self._last_infos[i])
                    continue
                msg = pipe.recv()
                if isinstance(msg, (tuple, list)) and len(msg) == 5:
                    obs, reward, terminated, truncated, info = msg
                    self._last_obs[i] = obs
                    self._last_rewards[i] = reward
                    self._last_terms[i] = terminated
                    self._last_truncs[i] = truncated
                    self._last_infos[i] = info
                else:
                    raise RuntimeError(f"Env {i} bad step message: {type(msg)} {msg}")
                self._awaiting[i] = False
            obs_list.append(self._last_obs[i])
            rewards.append(self._last_rewards[i])
            terms.append(self._last_terms[i])
            truncs.append(self._last_truncs[i])
            infos.append(self._last_infos[i])

        return (
            np.stack(obs_list),
            np.array(rewards),
            np.array(terms),
            np.array(truncs),
            infos,
        )

    def close(self):
        import signal
        import os
        # Try to close all pipes gracefully
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except Exception:
                pass
        # Wait for processes to exit, then force kill if needed
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                try:
                    # Try to kill the whole process group if possible
                    if hasattr(os, 'getpgid'):
                        try:
                            pgid = os.getpgid(p.pid)
                            os.killpg(pgid, signal.SIGKILL)
                        except Exception:
                            p.terminate()
                    else:
                        p.terminate()
                except Exception:
                    pass
        # Extra: kill any orphaned child processes from previous runs (Linux only)
        try:
            import psutil
            current_pid = os.getpid()
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
                if proc.info['ppid'] == current_pid and 'EmuHawk' in ' '.join(proc.info.get('cmdline', [])):
                    try:
                        proc.kill()
                    except Exception:
                        pass
        except ImportError:
            pass
