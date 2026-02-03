import gymnasium as gym
import numpy as np
import subprocess
import time
import os
import random

class MarioBizHawkRamEnv(gym.Env):
    """
    Environment using RAM-based observations only (no images).
    Observations: score, coins, lives, level, x_pos, state, timer, game_mode, frame, cmd_id
    """
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, rank=0, headless=False, frameskip=2, screenshot_every=2, obs_size=64, verbose=1):
        self._screenshot_every = screenshot_every
        self._obs_size = obs_size
        self._verbose = verbose
        super().__init__()
        self.rank = rank
        self.headless = headless
        self._frameskip = frameskip
        base_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\RAM\SNES"
        self._ram_file = os.path.join(base_path, f"smw_ram_{self.rank}.txt")
        self._action_file = os.path.join(base_path, f"smw_action_{self.rank}.txt")
        os.makedirs(base_path, exist_ok=True)
        if not os.path.exists(self._ram_file):
            with open(self._ram_file, "w") as f: f.write("0,0,0,0,0,0,0,0,0,0\n")
        self._command_id = 0
        self._write_action([0]*12)
        self._bizhawk_proc = None
        self._start_bizhawk()
        # Observation space: 10 RAM values
        self.observation_space = gym.spaces.Box(low=0, high=65535, shape=(10,), dtype=np.int32)
        self.action_space = gym.spaces.MultiBinary(12)
        self._last_score = None
        self._last_coins = None
        self._last_lives = None
        self._last_level = None
        self._last_x_pos = None
        self._last_emu_frame = -1
        self._stuck_steps = 0
        self._force_hard_reset = False
        self._restart_retry_interval_s = 60

    def _write_action(self, button_vector, is_reset=False):
        if is_reset:
            row = ["RESET"] + ["0"]*11 + [str(self._command_id)]
        else:
            row = [str(int(bit)) for bit in button_vector] + [str(self._command_id)]
        with open(self._action_file, "w") as f:
            f.write(",".join(row) + "\n")

    def _start_bizhawk(self):
        bizhawk_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\EmuHawk.exe"
        rom_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\Roms\Super Mario World (U) [!].smc"
        state_dir = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\SNES\State"
        savestates = [os.path.join(state_dir, f) for f in os.listdir(state_dir) if f.endswith('.State') and 'Super Mario World' in f]
        self._savestate_path = random.choice(savestates) if savestates else ""
        # Use the same Lua script as image env
        lua_script = os.path.join(os.path.dirname(self._ram_file), f"../Lua/SNES/smw_rl_control_{self.rank}.lua")
        cmd = [bizhawk_path, rom_path, "--lua", lua_script]
        if self._savestate_path: cmd += ["--load-state", self._savestate_path]
        startupinfo = None
        if self.headless:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        self._bizhawk_proc = subprocess.Popen(cmd, startupinfo=startupinfo)

    def _read_ram_once(self):
        if not os.path.exists(self._ram_file): return None
        try:
            with open(self._ram_file, "r") as f: line = f.readline()
            if not line: return None
            parts = line.strip().split(',')
            if len(parts) >= 10: return [int(parts[i]) for i in range(10)]
        except: pass
        return None

    def reset(self, *, seed=None, options=None):
        self._command_id += 1
        self._write_action([0]*12, is_reset=True)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            ram_data = self._read_ram_once()
            if ram_data and ram_data[-1] == self._command_id: break
            time.sleep(0.001)
        if ram_data is None: raise RuntimeError(f"Rank {self.rank}: Reset Timeout")
        self._last_score, self._last_coins, self._last_lives = ram_data[0], ram_data[1], ram_data[2]
        self._last_level, self._last_x_pos, self._last_emu_frame = ram_data[3], ram_data[4], ram_data[8]
        self._stuck_steps = 0
        return np.array(ram_data, dtype=np.int32), {}

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        obs = None
        for repeat in range(self._frameskip):
            self._command_id += 1
            self._write_action(action)
            deadline = time.time() + 2.0
            ram_data = None
            while time.time() < deadline:
                ram_data = self._read_ram_once()
                if ram_data and ram_data[-1] == self._command_id:
                    break
                time.sleep(0.001)
            if ram_data is None:
                if self._verbose:
                    print(
                        f"[Rank {self.rank}] RAM timeout at cmd_id={self._command_id}. "
                        f"Restarting BizHawk every {self._restart_retry_interval_s}s until it comes back."
                    )
                self._restart_bizhawk_with_retry(reason="ram_timeout")
                return obs, 0.0, False, False, {"error": "ram_timeout", "restarted": True}
            obs = np.array(ram_data, dtype=np.int32)
            score, coins, lives, level, x_pos, state, timer, game_mode, frame, cmd_id = ram_data
            if frame == self._last_emu_frame:
                self._stuck_steps += 1
            else:
                self._stuck_steps = 0
                self._last_emu_frame = frame
            if self._stuck_steps > (self._frameskip * 5):
                if self._verbose:
                    print(
                        f"[Rank {self.rank}] frame stuck. Restarting BizHawk every "
                        f"{self._restart_retry_interval_s}s until it comes back."
                    )
                self._restart_bizhawk_with_retry(reason="frame_stuck")
                return obs, 0.0, False, False, {"error": "frame_stuck", "restarted": True}
            reward = 0.0
            if self._last_x_pos is not None:
                progress = (x_pos - self._last_x_pos)
                if -100 < progress < 100:
                    reward += progress
            if self._last_coins is not None:
                coin_delta = coins - self._last_coins
                if coin_delta > 0:
                    reward += coin_delta * 10
            if self._last_score is not None:
                score_delta = score - self._last_score
                if score_delta > 0:
                    reward += score_delta * 0.2
            died = (self._last_lives is not None and lives < self._last_lives) or (state == 9) or (timer < 1)
            won = self._last_level is not None and level > self._last_level
            if not won and game_mode == 0x0E:
                died = True
            if died:
                reward -= 50
                done = True
            elif won:
                reward += 500
                done = True
            self._last_score, self._last_coins, self._last_lives = score, coins, lives
            self._last_level, self._last_x_pos = level, x_pos
            info = {"coins": coins, "level": level, "lives": lives, "x_pos": x_pos, "timer": timer, "score": score}
            total_reward += reward
            # Verbose logging
            if self._verbose == 2:
                print(f"Step {self._command_id}: obs={obs}, reward={reward}, info={info}")
            elif self._verbose == 1 and self._command_id % 100 == 0:
                print(f"Step {self._command_id}: reward={reward}, info={info}")
            if done:
                break
        return obs, total_reward, done, False, info

    def close(self):
        if hasattr(self, '_bizhawk_proc') and self._bizhawk_proc is not None:
            try: self._bizhawk_proc.terminate(); self._bizhawk_proc.wait(timeout=2)
            except: pass
            self._bizhawk_proc = None

    def _restart_bizhawk_with_retry(self, reason: str) -> None:
        while True:
            try:
                self.close()
                self._start_bizhawk()
                return
            except Exception as exc:
                if self._verbose:
                    print(
                        f"[Rank {self.rank}] BizHawk restart failed ({reason}): {exc}. "
                        f"Retrying in {self._restart_retry_interval_s}s..."
                    )
                time.sleep(self._restart_retry_interval_s)

if __name__ == "__main__":
    env = MarioBizHawkRamEnv(rank=0, headless=False)
    obs, _ = env.reset()
    print("Initial obs:", obs)
    for _ in range(10):
        obs, reward, done, _, info = env.step(env.action_space.sample())
        print(f"Obs: {obs}, Reward: {reward}")
    env.close()
