import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image
import sys

# Only import GUI automation modules on Windows
if os.name == "nt":
    import pyautogui
    import pygetwindow as gw
import subprocess
import time

class ZeldaSnesEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, rom_path, snes9x_path=None, core_path=None, retroarch_path='retroarch'):
        """
        rom_path: Path to the SNES ROM file
        snes9x_path: Path to Snes9x executable (Windows only)
        core_path: Path to SNES core for RetroArch (Linux/WSL only, e.g. '/usr/lib/libretro/snes9x_libretro.so')
        retroarch_path: Path to RetroArch executable (default: 'retroarch', assumes in PATH)
        """
        super().__init__()
        self.rom_path = rom_path
        self.snes9x_path = snes9x_path
        self.core_path = core_path or '/usr/lib/libretro/snes9x_libretro.so'
        self.retroarch_path = retroarch_path
        self.proc = None
        self.window = None
        # Example: 8 directions + A, B, X, Y, L, R, Start, Select
        self.action_space = spaces.Discrete(12)
        # Example: 256x224 SNES resolution, RGB
        self.observation_space = spaces.Box(low=0, high=255, shape=(224, 256, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        if self.proc:
            self.proc.terminate()
            time.sleep(1)
        if os.name == "nt":
            # Windows: use Snes9x
            self.proc = subprocess.Popen([self.snes9x_path, '-open', self.rom_path])
            # Wait for emulator window to appear (Windows only)
            max_wait = 20  # seconds
            waited = 0
            self.window = None
            while waited < max_wait:
                self.window = self._find_window()
                if self.window:
                    break
                time.sleep(0.5)
                waited += 0.5
            if not self.window:
                raise RuntimeError("Snes9x window not found after launch!")
            # Optional: wait a bit more for the game to be ready
            time.sleep(2)
        else:
            # Linux/WSL: use RetroArch with SNES core
            self.proc = subprocess.Popen([
                self.retroarch_path,
                '-L', self.core_path,
                self.rom_path
            ])
            self.window = None
            time.sleep(2)  # Just wait for emulator to start (Linux/WSL)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._send_action(action)
        time.sleep(0.05)
        obs = self._get_obs()
        reward = 0  # Placeholder
        done = False  # Placeholder
        info = {}
        return obs, reward, done, False, info

    def render(self, mode='human'):
        obs = self._get_obs()
        if mode == 'human':
            img = Image.fromarray(obs)
            img.show()
        return obs

    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc = None

    def _find_window(self):
        if os.name != "nt":
            return None
        # Print all window titles for debugging
        all_windows = gw.getAllTitles()
        print("[DEBUG] Open window titles:")
        for title in all_windows:
            print(f"  - {title}")
        windows = gw.getWindowsWithTitle('Snes9x 1.60')
        if windows:
            win = windows[0]
            win.activate()
            return win
        return None

    def _get_obs(self):
        if os.name == "nt" and self.window:
            bbox = (self.window.left, self.window.top, self.window.right, self.window.bottom)
            img = pyautogui.screenshot(region=bbox)
            img = img.resize((256, 224))
            return np.array(img)
        else:
            # On Linux/WSL, return a blank observation or implement another method
            return np.zeros((224, 256, 3), dtype=np.uint8)

    def _send_action(self, action):
        # Map action index to key
        keys = ['up', 'down', 'left', 'right', 'z', 'x', 'a', 's', 'q', 'w', 'enter', 'space']
        if os.name == "nt" and 0 <= action < len(keys):
            pyautogui.press(keys[action])
        else:
            # On Linux/WSL, do nothing or implement alternative action sending
            pass
