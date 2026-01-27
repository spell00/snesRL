import gymnasium as gym
import numpy as np
import subprocess
import time
import os
import random
import cv2
import hashlib


# --- RewardSMW helper class ---
class RewardSMW:
    def __init__(
        self,
        progress_scale: float = 0.01,
        win_bonus: float = 1000.0,
        death_penalty: float = -50.0,
        stuck_steps: int = 90,
        stuck_penalty: float = 0.0,
        use_novelty: bool = True,
        novelty_bonus: float = 0.01,
        novelty_max_items: int = 200_000,
        clamp_negative_progress: bool = True,
    ):
        self.progress_scale = float(progress_scale)
        self.win_bonus = float(win_bonus)
        self.death_penalty = float(death_penalty)
        self.stuck_steps = int(stuck_steps)
        self.stuck_penalty = float(stuck_penalty)

        self.use_novelty = bool(use_novelty)
        self.novelty_bonus = float(novelty_bonus)
        self.novelty_max_items = int(novelty_max_items)

        self.clamp_negative_progress = bool(clamp_negative_progress)
        self.reset()

    def reset(self):
        self.best_progress = None
        self.prev_progress = None
        self.steps_since_progress = 0
        self.seen_hashes = set()

    def _extract_progress(self, ram: dict) -> float:
        x = ram.get("x_pos", None)
        if x is None:
            raise KeyError("RewardSMW needs ram['x_pos'] (or override _extract_progress).")
        return float(x)

    def _is_win(self, ram: dict, info: dict | None = None) -> bool:
        return bool(info and info.get("win", False))

    def _is_death(self, ram: dict, info: dict | None = None) -> bool:
        return bool(info and info.get("death", False))

    def step_reward(
        self,
        obs_img: np.ndarray | None,
        ram: dict,
        terminated: bool,
        truncated: bool = False,
        info: dict | None = None,
    ) -> float:
        r = 0.0

        progress = self._extract_progress(ram)
        if self.best_progress is None:
            self.best_progress = progress
            self.prev_progress = progress
        else:
            delta_best = progress - self.best_progress
            if self.clamp_negative_progress and delta_best < 0:
                delta_best = 0.0

            if delta_best > 0:
                r += delta_best * self.progress_scale
                self.best_progress = progress
                self.steps_since_progress = 0
            else:
                self.steps_since_progress += 1

            self.prev_progress = progress

        if self.stuck_steps > 0 and self.steps_since_progress >= self.stuck_steps:
            r += self.stuck_penalty
            self.steps_since_progress = 0

        if self.use_novelty and obs_img is not None and len(self.seen_hashes) < self.novelty_max_items:
            h = hashlib.blake2b(obs_img.tobytes(), digest_size=8).hexdigest()
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                r += self.novelty_bonus

        done = bool(terminated or truncated)
        if done:
            if self._is_win(ram, info):
                r += self.win_bonus
            elif self._is_death(ram, info):
                r += self.death_penalty

        return float(r)


class MarioBizHawkEnv(gym.Env):
    """
    BizHawk + Lua file-IPC environment.
    Obs: 84x84 grayscale (bmp screenshot)
    Act: Discrete or MultiBinary(8)
    IPC:
      - Python writes ctrl file (ACT/LOAD/SAVE + cmd_id)
      - Lua writes ram file "...,cmd_id" as ack
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def _atomic_write_line(self, path: str, line: str):
        tmp = path + ".tmp"
        with open(tmp, "w", newline="\n") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    # Savestate auto-save logic disabled: always reload original state on reset.
    def maybe_auto_save_savestate(self, x_pos):
        pass

    def __init__(
        self,
        rank=0,
        headless=False,
        frameskip=2,
        screenshot_every=30,
        obs_size=84,
        verbose=1,
        reset_mode="soft",
        state_dir=None,
        ram_timeout_s=10.0, # Increased from 4.0 to handle training lag
        exploration_dir=None,
        reset_timeout_s=15.0,
        startup_sleep_s=4.0,
        progress_per_pixel=0.0,
        exploration_bonus=1.0,
        coin_reward=5.0,
        score_reward=0.01,
        death_penalty=-500.0,
        win_bonus=500.0,
        stuck_penalty=0.0,
        normalize_progress_by_frameskip=True,
        seed=None,
        novelty_enabled=True,
        enable_cell_exploration=False,
        model_name="default",
        action_type="discrete",
        cell_bonus_mode="linear",
    ):
        super().__init__()
        self.rank = int(rank)
        self._min_y_pos = None

        if seed is None:
            seed = 12345 + self.rank
        self._seed = int(seed)
        random.seed(self._seed)
        np.random.seed(self._seed)

        self.headless = bool(headless)
        self._frameskip = int(frameskip)
        self._screenshot_every = int(screenshot_every)
        self._obs_size = int(obs_size)
        self._verbose = int(verbose)
        self._reset_mode = str(reset_mode)

        self._ram_timeout_s = float(ram_timeout_s)
        self._reset_timeout_s = float(reset_timeout_s)
        self._startup_sleep_s = float(startup_sleep_s)

        self._progress_per_pixel = float(progress_per_pixel)
        self._coin_reward = float(coin_reward)
        self._score_reward = float(score_reward)
        self._death_penalty = float(death_penalty)
        self._win_bonus = float(win_bonus)
        self._stuck_penalty = float(stuck_penalty)
        self._normalize_progress_by_frameskip = bool(normalize_progress_by_frameskip)

        self._exploration_bonus = float(exploration_bonus)
        self._novelty_enabled = bool(novelty_enabled)
        self._enable_cell_exploration = bool(enable_cell_exploration)
        self._cell_bonus_mode = str(cell_bonus_mode)

        self._stuck_max_steps = 3000
        self._stuck_counter = 0
        self._consecutive_timeouts = 0 # Track timeouts for auto-restart

        # --- No progress timeout tracking ---
        self._no_progress_timeout_s = 30.0  # 30 seconds in-game time
        self._no_progress_start_x = None
        self._no_progress_start_y = None
        self._no_progress_start_frame = None
        self._no_progress_last_progress = False
        self._no_progress_reset_pending = False

        # Action space
        self._action_type = action_type
        if self._action_type == "discrete":
            # Button order: Right, Left, Up, Down, Y (run), B (jump), A, X
            # Always hold Y (run), never use Start, Select, L, R
            # No more than 2 directions, no opposites
            directions = [
                (1, 0, 0, 0),  # Right
                (0, 1, 0, 0),  # Left
                (0, 0, 1, 0),  # Up
                (0, 0, 0, 1),  # Down
                (1, 0, 1, 0),  # Right + Up
                (1, 0, 0, 1),  # Right + Down
                (0, 1, 1, 0),  # Left + Up
                (0, 1, 0, 1),  # Left + Down
                (0, 0, 0, 0),  # No direction
            ]
            actions = []
            for dir in directions:
                # Always hold Y (run)
                base = list(dir) + [1, 0, 0, 0]
                actions.append(tuple(base))  # No jump
                # Add jump (B)
                base_jump = list(dir) + [1, 1, 0, 0]
                actions.append(tuple(base_jump))
            self._discrete_actions = actions
            print(actions)
            self.action_space = gym.spaces.Discrete(len(self._discrete_actions))
        elif self._action_type == "multibinary":
            self.action_space = gym.spaces.MultiBinary(8)
        elif self._action_type == "box":
            self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown action_type: {self._action_type}")

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self._obs_size, self._obs_size, 1), dtype=np.uint8
        )

        # Paths
        base_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\RAM\SNES"
        os.makedirs(base_path, exist_ok=True)
        self._ram_file = os.path.join(base_path, f"smw_ram_{self.rank}.txt")
        self._ctrl_file = os.path.join(base_path, f"smw_ctrl_{self.rank}.txt")
        self._screenshot_file = os.path.join(base_path, f"smw_screen_{self.rank}.bmp")

        self._state_dir = state_dir or r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\SNES\State"

        self._exploration_dir = exploration_dir or os.path.join(
            os.path.dirname(__file__), f"explore_{model_name}_rank{self.rank}"
        )
        os.makedirs(self._exploration_dir, exist_ok=True)
        self._explored_hashes = set()
        # For cell exploration
        self._cell_bins = (128, 64)
        self._explored_cells = set()
        self._explored_hashes = set()
        self._cell_origin_x = None
        self._cell_origin_y = None

        # Ensure RAM file exists
        if not os.path.exists(self._ram_file):
            with open(self._ram_file, "w") as f:
                f.write("0,0,0,0,0,0,0,0,0,0,0\n")

        # Command id + initial ctrl
        self._command_id = 0
        # Use a valid default action for each action type
        if self._action_type == "discrete":
            default_action = 0
        elif self._action_type == "multibinary":
            default_action = [0] * 8
        elif self._action_type == "box":
            default_action = [0.0] * 8
        else:
            raise ValueError(f"Unknown action_type: {self._action_type}")
        self._write_act(default_action, cmd_id=self._command_id)

        # Observation cache
        self._last_obs = np.zeros((self._obs_size, self._obs_size, 1), dtype=np.uint8)
        self._last_screenshot_mtime = 0.0

        # Trackers
        self._last_score = None
        self._last_coins = None
        self._last_lives = None
        self._last_level = None
        self._last_x_pos = None

        self._last_emu_frame = -1
        self._frame_stuck_steps = 0

        self._bizhawk_proc = None
        self._start_bizhawk()

        self.rewarder = RewardSMW(
            progress_scale=0.01,
            win_bonus=1000.0,
            death_penalty=-50.0,
            stuck_steps=600,
            stuck_penalty=self._stuck_penalty,
            use_novelty=True,
            novelty_bonus=0.01,
            novelty_max_items=200_000,
            clamp_negative_progress=True,
        )

    # -------- IPC writers --------

    # Disabled: do not auto-save progress savestates
    def autosave_progress_state(self, x_pos: int, level: int | None = None) -> str:
        return ""

    def _write_ctrl_line(self, line: str):
        for _ in range(50):
            try:
                with open(self._ctrl_file, "w", newline="\n") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                return
            except PermissionError:
                time.sleep(0.002)
        # last try without fsync
        with open(self._ctrl_file, "w", newline="\n") as f:
            f.write(line)


    def _write_act(self, action, cmd_id=None):
        if cmd_id is None:
            cmd_id = self._command_id
        if self._action_type == "discrete":
            button_vector = self._discrete_actions[action]
        elif self._action_type == "multibinary":
            button_vector = action
        elif self._action_type == "box":
            button_vector = [1 if float(b) >= 0.5 else 0 for b in action]
        else:
            raise ValueError(f"Unknown action_type: {self._action_type}")

        row = ["ACT"] + [str(int(b)) for b in button_vector] + [str(int(cmd_id))]
        self._write_ctrl_line(",".join(row) + "\n")

    def _write_load(self, savestate_path, cmd_id=None):
        if cmd_id is None:
            cmd_id = self._command_id
        p = savestate_path.replace("\\", "/")
        row = ["LOAD", p, str(int(cmd_id))]
        self._write_ctrl_line(",".join(row) + "\n")

    def save_savestate(self, savestate_path, cmd_id=None):
        if cmd_id is None:
            cmd_id = self._command_id + 1
            self._command_id = cmd_id
        p = savestate_path.replace("\\", "/")
        row = ["SAVE", p, str(int(cmd_id))]
        self._write_ctrl_line(",".join(row) + "\n")
        ack = self._wait_for_cmd_ack(cmd_id, timeout_s=self._reset_timeout_s)
        if ack is None:
            raise RuntimeError(f"Rank {self.rank}: SAVE timeout (cmd_id={cmd_id})")
        return ack

    # -------- Lua script generation --------
    def _create_lua_script(self):
        lua_dir = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\Lua\SNES"
        os.makedirs(lua_dir, exist_ok=True)
        target_lua = os.path.join(lua_dir, f"smw_rl_control_{self.rank}.lua")

        lua_ram = self._ram_file.replace("\\", "/")
        lua_ctrl = self._ctrl_file.replace("\\", "/")
        lua_screenshot = self._screenshot_file.replace("\\", "/")

        lua_content = f'''\
local ctrl_file = "{lua_ctrl}"
local ram_file = "{lua_ram}"
local screenshot_file = "{lua_screenshot}"

local frames_to_advance = {int(self._frameskip)}
local screenshot_every = {int(self._screenshot_every)}
local last_processed_id = -1

local buttons_map = {{
  "P1 Right", "P1 Left", "P1 Up", "P1 Down",
  "P1 Y", "P1 B", "P1 A", "P1 X"
}}

local heartbeat_every = 60
local last_heartbeat = 0

function write_ram(cmd_id)
    local score = memory.read_u16_le(0x0F34)
    local coins = memory.read_u8(0x0DBF)
    local lives = memory.read_u8(0x0DBE)
    local level = memory.read_u8(0x13BF)
    local x_pos = memory.read_u16_le(0x0094)
    local y_pos = memory.read_u16_le(0x0096)
    local state = memory.read_u8(0x0071)
    local timer = memory.read_u8(0x0F31)*100 + memory.read_u8(0x0F32)*10 + memory.read_u8(0x0F33)
    local game_mode = memory.read_u8(0x0100)
    local frame = emu.framecount()

    local line = string.format("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\\n",
        score, coins, lives, level, x_pos, y_pos, state, timer, game_mode, frame, cmd_id)

    local f = io.open(ram_file, "w")
    if f then
        f:write(line)
        f:flush()
        f:close()
    end
end

local function maybe_screenshot(cmd_id, force)
  if force or cmd_id == 0 or (cmd_id % screenshot_every == 0) then
    pcall(function() client.screenshot(screenshot_file) end)
  end
end

local function read_ctrl_line()
  local f = io.open(ctrl_file, "r")
  if not f then return nil end
  local line = f:read("*l")
  f:close()
  if not line or line == "" then return nil end
  return line
end

local function process_command()
  local line = read_ctrl_line()
  if not line then return false end

  local parts = {{}}
  for val in string.gmatch(line, "([^,]+)") do
    table.insert(parts, val)
  end
  if #parts < 2 then return false end

  local cmd_type = parts[1]
  local cmd_id = tonumber(parts[#parts])
  if not cmd_id then return false end
  if cmd_id <= last_processed_id then return false end
  last_processed_id = cmd_id

  if cmd_type == "LOAD" then
    local p = parts[2]
    if p and p ~= "" then savestate.load(p) end
    emu.frameadvance()
    maybe_screenshot(cmd_id, true)
    write_ram(cmd_id)
    return true
  end

  if cmd_type == "SAVE" then
    local p = parts[2]
    if p and p ~= "" then savestate.save(p) end
    emu.frameadvance()
    maybe_screenshot(cmd_id, true)
    write_ram(cmd_id)
    return true
  end

  if cmd_type == "ACT" then
    local btns = {{}}
    for i=1,8 do
      local bit = parts[i+1]
      btns[buttons_map[i]] = (bit == "1")
    end

    if btns["P1 Left"] and btns["P1 Right"] then btns["P1 Left"] = false end
    if btns["P1 Up"] and btns["P1 Down"] then btns["P1 Up"] = false end

    for i=1,frames_to_advance do
      joypad.set(btns)
      emu.frameadvance()
    end

    maybe_screenshot(cmd_id, false)
    write_ram(cmd_id)
    return true
  end

  write_ram(cmd_id)
  return true
end

while true do
  local did = process_command()
  if not did then
    local fr = emu.framecount()
    if fr - last_heartbeat >= heartbeat_every then
      last_heartbeat = fr
      write_ram(last_processed_id)
    end
    emu.yield()
  end
end
'''
        with open(target_lua, "w", newline="\n") as f:
            f.write(lua_content)
        return target_lua

    # -------- BizHawk start/stop --------
    def _start_bizhawk(self):
        bizhawk_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\EmuHawk.exe"
        rom_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\Roms\Super Mario World (U) [!].smc"

        lua_script = self._create_lua_script()

        if self._verbose >= 1:
            print(f"--- [Rank {self.rank}] Starting BizHawk (frameskip={self._frameskip}) ---")

        cmd = [bizhawk_path, rom_path, "--lua", lua_script]

        startupinfo = None
        if self.headless:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        self._bizhawk_proc = subprocess.Popen(cmd, startupinfo=startupinfo)
        time.sleep(self._startup_sleep_s)

    def close(self):
        if getattr(self, "_bizhawk_proc", None) is not None:
            try:
                self._bizhawk_proc.terminate()
                self._bizhawk_proc.wait(timeout=2)
            except Exception:
                pass
            self._bizhawk_proc = None

    # -------- Helpers --------
    def _read_ram_once(self):
        if not os.path.exists(self._ram_file):
            return None
        try:
            with open(self._ram_file, "r") as f:
                line = f.readline()
            if not line:
                return None
            parts = line.strip().split(",")
            if len(parts) < 11:
                return None
            return [int(parts[i]) for i in range(11)]
        except Exception:
            return None

    def _wait_for_cmd_ack(self, cmd_id, timeout_s):
        deadline = time.time() + float(timeout_s)
        last = None
        start_mtime = os.path.getmtime(self._ram_file) if os.path.exists(self._ram_file) else 0.0

        while time.time() < deadline:
            last = self._read_ram_once()
            if last is not None and int(last[-1]) == int(cmd_id):
                return last

            # If RAM file never changes, we are not getting any new data
            if os.path.exists(self._ram_file):
                mtime = os.path.getmtime(self._ram_file)
                if mtime > start_mtime:
                    start_mtime = mtime  # it is changing, keep waiting

            time.sleep(0.002)

        return None

    def _get_obs(self):
        if not os.path.exists(self._screenshot_file):
            return self._last_obs
        try:
            mtime = os.path.getmtime(self._screenshot_file)
            if mtime == self._last_screenshot_mtime:
                return self._last_obs
            img = cv2.imread(self._screenshot_file, 0)
            if img is None:
                return self._last_obs
            if img.shape[0] != self._obs_size or img.shape[1] != self._obs_size:
                img = cv2.resize(img, (self._obs_size, self._obs_size), interpolation=cv2.INTER_NEAREST)
            self._last_obs = img.reshape((self._obs_size, self._obs_size, 1))
            self._last_screenshot_mtime = mtime
            return self._last_obs
        except Exception:
            return self._last_obs

    def _list_savestates(self):
        if not os.path.isdir(self._state_dir):
            return []
        states = []
        for f in os.listdir(self._state_dir):
            if f.endswith(".State") and "SMW" in f:
                states.append(os.path.join(self._state_dir, f))
        return sorted(states)

    def get_savestate_by_index(self, idx: int):
        states = self._list_savestates()
        if not states:
            raise RuntimeError(f"No savestates found in {self._state_dir}")
        if idx == 0:
            return None
        if idx < 1 or idx > len(states):
            raise ValueError(f"Level index {idx} out of range (1-{len(states)})")
        return states[idx - 1]

    # -------- Gym API --------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._reset_mode == "hard":
            self.close()
            self._start_bizhawk()

        # ✅ Reset episode-level exploration memory on every reset
        self._explored_cells = set()
        self._explored_hashes = set()
        self._stuck_counter = 0
        self._min_y_pos = None
        self._cell_origin_x = None
        self._cell_origin_y = None

        states = self._list_savestates()
        if not states:
            raise RuntimeError(f"No savestates found in {self._state_dir}")

        chosen = options.get("savestate") if options and options.get("savestate") else random.choice(states)

        self._command_id += 1
        self._write_load(chosen, cmd_id=self._command_id)

        ram_data = self._wait_for_cmd_ack(self._command_id, timeout_s=self._reset_timeout_s)
        if ram_data is None:
            raise RuntimeError(f"Rank {self.rank}: reset LOAD timeout (cmd_id={self._command_id})")

        score, coins, lives, level, x_pos, y_pos, state, timer, game_mode, frame, cmd_id = ram_data

        self._last_score = score
        self._last_coins = coins
        self._last_lives = lives
        self._last_level = level
        self._last_x_pos = x_pos
        self._last_y_pos = y_pos
        self._last_emu_frame = frame
        self._frame_stuck_steps = 0
        self._stuck_counter = 0

        self.rewarder.reset()

        # Track for no-progress timeout
        self._no_progress_start_x = x_pos
        self._no_progress_start_y = y_pos
        self._no_progress_start_frame = frame
        self._no_progress_last_progress = False
        self._no_progress_reset_pending = False

        obs = self._get_obs()
        info = {"savestate": chosen, "level": level, "x_pos": x_pos, "y_pos": y_pos, "cmd_id": cmd_id}
        return obs, info

    def step(self, action):
        self._command_id += 1
        self._write_act(action, cmd_id=self._command_id)

        ram_data = self._wait_for_cmd_ack(self._command_id, timeout_s=self._ram_timeout_s)
        if ram_data is None:
            self._consecutive_timeouts += 1
            if self._verbose >= 1:
                print(f"[Rank {self.rank}] RAM timeout at cmd_id={self._command_id} ({self._consecutive_timeouts}/3).")
            
            # If we hit 3 timeouts in a row, the emulator is likely dead - restart it
            if self._consecutive_timeouts >= 3:
                print(f"[Rank {self.rank}] Too many timeouts; restarting BizHawk.")
                self.close()
                self._start_bizhawk()
                self._consecutive_timeouts = 0
            
            obs = self._get_obs()
            return obs, 0.0, False, False, {"error": "ram_timeout"}

        self._consecutive_timeouts = 0 # Reset on success

        obs = self._get_obs()
        score, coins, lives, level, x_pos, y_pos, state, timer, game_mode, frame, cmd_id = ram_data

        # Frame stuck detection
        if frame == self._last_emu_frame:
            self._frame_stuck_steps += 1
        else:
            self._frame_stuck_steps = 0
            self._last_emu_frame = frame

        if self._frame_stuck_steps > (self._frameskip * 5):
            if self._verbose >= 1:
                print(f"[Rank {self.rank}] frame stuck; restarting BizHawk.")
            self.close()
            self._start_bizhawk()
            obs = self._get_obs()
            return obs, 0.0, False, False, {"error": "frame_stuck_restart"}

        # --- No progress timeout logic ---
        # If no progress (x or y change) for 30 seconds, reset.
        no_progress = False
        if self._last_x_pos is not None and self._last_y_pos is not None:
            if x_pos != self._last_x_pos or y_pos != self._last_y_pos:
                # Mario moved, reset the timer
                self._no_progress_start_frame = frame
            
            elapsed_frames = frame - self._no_progress_start_frame
            if elapsed_frames >= int(self._no_progress_timeout_s * 60):
                no_progress = True

        if no_progress:
            if self._verbose >= 1:
                print(f"[Rank {self.rank}] No progress for 30 seconds; resetting.")
            # Reset tracking for next episode
            self._no_progress_start_frame = frame 
            return obs, 0.0, False, True, {"error": "no_progress_timeout"}

        ram_dict = {
            "score": score,
            "coins": coins,
            "lives": lives,
            "level": level,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "state": state,
            "timer": timer,
            "game_mode": game_mode,
            "frame": frame,
        }

        info_flags = {
            "win": (self._last_level is not None and level > self._last_level),
            "death": (
                (self._last_lives is not None and lives < self._last_lives)
                or timer <= 0
                or state == 9
                or game_mode == 0x0E
            ),
        }

        # Reward is based on score delta
        reward = 0.0
        if self._last_score is not None:
            score_delta = score - self._last_score
            reward += score_delta * self._score_reward
        # reward = self.rewarder.step_reward(obs, ram_dict, terminated=False, truncated=False, info=info_flags)
        # Cell-based exploration bonus (discretize x/y into bins)
        if self._enable_cell_exploration:
            # On first frame, store Mario's initial position as origin
            if self._cell_origin_x is None or self._cell_origin_y is None:
                self._cell_origin_x = x_pos
                self._cell_origin_y = y_pos
            # if y_pos - self._cell_origin_y < 0:
            #     self._cell_origin_y = y_pos  # Adjust origin if going upwards
            rel_x = abs(x_pos - self._cell_origin_x)
            rel_y = abs(y_pos - self._cell_origin_y)
            bin_x = int(rel_x // self._cell_bins[0])
            bin_y = int(rel_y // self._cell_bins[1])
            cell = (bin_x, bin_y)
            # Print when new minimum y_pos is reached (i.e., Mario jumps higher)
            if self._min_y_pos is None or y_pos < self._min_y_pos:
                self._min_y_pos = y_pos
                # print(f"[Rank {self.rank}] New min y_pos: {y_pos}")
            new_cell = cell not in self._explored_cells
            if new_cell:
                self._explored_cells.add(cell)
                if self._cell_bonus_mode == "linear":
                    # Linear distance from origin (0,0)
                    dist = ((bin_x ** 2 + bin_y ** 2) ** 0.5)
                    reward += dist * self._exploration_bonus
                elif self._cell_bonus_mode == "constant":
                    reward += self._exploration_bonus
                elif self._cell_bonus_mode == "quadratic":
                    dist = ((bin_x ** 2 + bin_y ** 2) ** 2)
                    reward += dist * self._exploration_bonus
                else:
                    dist = (bin_x ** 2 + bin_y ** 2)
                    reward += dist * self._exploration_bonus
                if self._verbose >= 2:
                    print(f"[Rank {self.rank}] New cell explored: {cell}, total cells: {len(self._explored_cells)}")
            # No bonus for revisiting cells
        elif self._novelty_enabled:
            # Cheap novelty bonus (legacy)
            obs_hash = hashlib.blake2b(obs.tobytes(), digest_size=8).hexdigest()
            if obs_hash not in self._explored_hashes:
                self._explored_hashes.add(obs_hash)
                reward += self._exploration_bonus
                self._stuck_counter = 0
            else:
                self._stuck_counter += 1

        if reward > 0:
            pass
        died = False
        won = False
        timeout_death = False

        if self._last_lives is not None and lives < self._last_lives:
            died = True
        if timer <= 0:
            died = True
            timeout_death = True
        if state == 9 or game_mode == 0x0E:
            died = True
        if self._last_level is not None and level > self._last_level:
            won = True

        terminated = False
        truncated = False
        if died:
            terminated = True
            reward += self._death_penalty if not timeout_death else self._death_penalty
        elif won:
            terminated = True
            reward += self._win_bonus
        elif self._stuck_counter >= self._stuck_max_steps:
            truncated = True
            self._stuck_counter = 0
            reward += self._death_penalty
            if self._verbose >= 1:
                print(f"[Rank {self.rank}] Episode truncated due to stuck detection.")

        self._last_score = score
        self._last_coins = coins
        self._last_lives = lives
        self._last_level = level
        self._last_x_pos = x_pos
        # Reset explored cells on episode end
        if terminated or truncated:
            self._explored_cells = set()
            self._explored_hashes = set()

        # Auto-save savestate logic disabled
        # self.maybe_auto_save_savestate(x_pos)

        info = {
            "score": score,
            "coins": coins,
            "lives": lives,
            "level": level,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "timer": timer,
            "state": state,
            "game_mode": game_mode,
            "frame": frame,
            "cmd_id": cmd_id,
        }
        info["win"] = won
        info["death"] = died
        return obs, float(reward), bool(terminated), bool(truncated), info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()

    # Determine use_novelty value
    if args.use_novelty is not None:
        use_novelty = True
    elif args.no_use_novelty:
        use_novelty = False
    else:
        use_novelty = True

    env = MarioBizHawkEnv()
    env.rewarder.use_novelty = use_novelty
    if args.level == 0:
        obs, info = env.reset()
    else:
        savestate = env.get_savestate_by_index(args.level)
        obs, info = env.reset(options={"savestate": savestate})
    print(f"Loaded level: {info['level']}, savestate: {info['savestate']}, use_novelty: {env.rewarder.use_novelty}")
