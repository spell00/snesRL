import gymnasium as gym
import numpy as np
import subprocess
import time
import os
import random

class ZeldaBizHawkEnv(gym.Env):
    """
    Gymnasium environment for Zelda ALttP using BizHawk + Lua RAM file.
    Actions: currently placeholder (extend for real input)
    Observations: rupees, hearts, max_hearts, room_id
    Reward: change in rupees
    """
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)  # Placeholder, extend for real actions
        self.observation_space = gym.spaces.Box(low=0, high=999, shape=(4,), dtype=np.int32)
        self._last_rupees = None
        self._last_hearts = None
        self._ram_file = "zelda_ram.txt"
        self._start_bizhawk()

    def _start_bizhawk(self):
        # Launch BizHawk with Super Mario World ROM, Lua script, and load a random save state
        bizhawk_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\EmuHawk.exe"
        rom_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\Roms\Super Mario World (U) [!].smc"
        lua_script = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\Lua\SNES\smw_ram_file.lua"
        state_dir = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\SNES\State"
        savestates = [os.path.join(state_dir, f) for f in os.listdir(state_dir) if f.endswith(".State")]
        if savestates:
            savestate_path = random.choice(savestates)
        else:
            savestate_path = None
        cmd = [bizhawk_path, rom_path, "--lua", lua_script]
        if savestate_path:
            cmd += ["--load-state", savestate_path]
        subprocess.Popen(cmd)

    def reset(self, *, seed=None, options=None):
        obs = self._get_obs()
        self._last_rupees = obs[0]
        self._last_hearts = obs[1]
        return obs, {}

    def step(self, action):
        obs = self._get_obs()
        print(f"Obs: {obs}")  # Debug: print every observation
        reward = obs[0] - (self._last_rupees if self._last_rupees is not None else obs[0])
        # Print score when a heart is picked up
        if self._last_hearts is not None and obs[1] > self._last_hearts:
            print(f"Heart picked up! Hearts: {self._last_hearts} -> {obs[1]}")
        self._last_rupees = obs[0]
        self._last_hearts = obs[1]
        done = False  # Extend for episode logic
        info = {}
        return obs, reward, done, False, info

    def _get_obs(self):
        # Wait for zelda_ram.txt to exist and have data
        for _ in range(100):
            if os.path.exists(self._ram_file):
                with open(self._ram_file, "r") as f:
                    line = f.readline()
                    if line:
                        try:
                            rupees, hearts, max_hearts, room_id = map(int, line.strip().split(','))
                            return np.array([rupees, hearts, max_hearts, room_id], dtype=np.int32)
                        except Exception as e:
                            print("Error parsing RAM file line:", line, e)
            time.sleep(0.1)
        raise RuntimeError("Could not read RAM values from file")

    def close(self):
        pass

# Example usage:
if __name__ == "__main__":
    env = ZeldaBizHawkEnv()
    obs, _ = env.reset()
    print("Initial obs:", obs)
    for _ in range(10):
        obs, reward, done, _, info = env.step(0)
        print(f"Obs: {obs}, Reward: {reward}")
    env.close()
