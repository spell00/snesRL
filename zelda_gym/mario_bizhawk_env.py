import atexit
import hashlib
import os
import random
import shlex
import signal
import subprocess
import time
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np



# ---------------- Reward helper ----------------
class RewardSMW:
    def __init__(
        self,
        progress_scale: float = 0.01,
        win_bonus: float = 1000.0,
        death_penalty: float = -5.0,
        stuck_steps: int = 90,
        stuck_penalty: float = 0.0,
        use_novelty: bool = True,
        novelty_bonus: float = 0.01,
        novelty_max_items: int = 200_000,
        clamp_negative_progress: bool = True,
        enable_cell_exploration: bool = False,
        cell_bonus_mode: str = "constant",
        exploration_bonus: float = 1.0,
        cell_bins: tuple[int, int] = (64, 64),
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
        
        self.enable_cell_exploration = bool(enable_cell_exploration)
        self.cell_bonus_mode = str(cell_bonus_mode)
        self.exploration_bonus = float(exploration_bonus)
        self.cell_bins = cell_bins
        
        self.reset()

    def reset(self):
        self.best_progress = None
        self.steps_since_progress = 0
        self.seen_hashes = set()
        
        self._explored_cells = set()
        self._cell_origin_x = None
        self._cell_origin_y = None
        self._min_y_pos = None

    def _extract_progress(self, ram: dict) -> float:
        x = ram.get("x_pos", None)
        if x is None:
            raise KeyError("RewardSMW needs ram['x_pos'].")
        return float(x)

    def step_reward(
        self,
        obs_img: Optional[np.ndarray],
        ram: dict,
        terminated: bool,
        truncated: bool = False,
        info: Optional[dict] = None,
    ) -> tuple[float, dict]:
        r = 0.0
        extra_info = {
            "reward_progress": 0.0,
            "reward_novelty": 0.0,
            "reward_cell": 0.0,
            "reward_win": 0.0,
            "reward_death": 0.0,
            "reward_stuck": 0.0,
            "new_progress": False,
            "new_novelty": False,
            "new_cell": False,
        }

        # 1. Progress
        progress = self._extract_progress(ram)
        if self.best_progress is None:
            self.best_progress = progress
        else:
            delta = progress - self.best_progress
            if self.clamp_negative_progress and delta < 0:
                delta = 0.0
            if delta > 0:
                delta_r = delta * self.progress_scale
                r += delta_r
                extra_info["reward_progress"] = delta_r
                self.best_progress = progress
                self.steps_since_progress = 0
                extra_info["new_progress"] = True
            else:
                self.steps_since_progress += 1

        if self.stuck_steps > 0 and self.steps_since_progress >= self.stuck_steps:
            r += self.stuck_penalty
            extra_info["reward_stuck"] = self.stuck_penalty
            self.steps_since_progress = 0

        # 2. Novelty
        if self.use_novelty and obs_img is not None and len(self.seen_hashes) < self.novelty_max_items:
            h = hashlib.blake2b(obs_img.tobytes(), digest_size=8).hexdigest()
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                r += self.novelty_bonus
                extra_info["reward_novelty"] = self.novelty_bonus
                extra_info["new_novelty"] = True
        
        # 3. Cell Exploration
        if self.enable_cell_exploration:
            x_pos = ram.get("x_pos")
            y_pos = ram.get("y_pos")
            
            if x_pos is not None and y_pos is not None:
                if self._cell_origin_x is None or self._cell_origin_y is None:
                    self._cell_origin_x = x_pos
                    self._cell_origin_y = y_pos
                    
                rel_x = abs(x_pos - self._cell_origin_x)
                rel_y = abs(y_pos - self._cell_origin_y)
                bin_x = int(rel_x // self.cell_bins[0])
                bin_y = int(rel_y // self.cell_bins[1])
                cell = (bin_x, bin_y)
                
                if self._min_y_pos is None or y_pos < self._min_y_pos:
                    self._min_y_pos = y_pos

                if cell not in self._explored_cells:
                    self._explored_cells.add(cell)
                    extra_info["new_cell"] = True
                    cell_r = 0.0
                    if self.cell_bonus_mode == "constant":
                        cell_r = self.exploration_bonus
                    else:
                        dist = (bin_x**2 + bin_y**2) ** 0.5
                        cell_r = dist * self.exploration_bonus
                    r += cell_r
                    extra_info["reward_cell"] = cell_r

        # 4. Terminal
        done = bool(terminated or truncated)
        if done and info:
            if info.get("win", False):
                r += self.win_bonus
                extra_info["reward_win"] = self.win_bonus
            elif info.get("death", False):
                r += self.death_penalty
                extra_info["reward_death"] = self.death_penalty

        return float(r), extra_info


# ---------------- BizHawk Env ----------------
class MarioBizHawkEnv(gym.Env):

    @staticmethod
    def discrete_action_count():
        directions = [
            (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1),
            (1,0,1,0), (1,0,0,1), (0,1,1,0), (0,1,0,1), (0,0,0,0)
        ]
        return len(directions) * 2
    """
    BizHawk + Lua file-IPC environment.

    IPC:
      - Python writes ctrl file: ACT/LOAD/SAVE + cmd_id
      - Lua writes ram file: "... , cmd_id" as ack
      - Lua writes -888 as handshake marker once it starts.

    Linux headless:
      - uses xvfb-run + openbox so xdotool can activate modal dialogs
      - starts BizHawk exactly once in __init__
      - close() kills whole process group
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # -------- small IO utils --------
    @staticmethod
    def _atomic_write_line(path: str, line: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", newline="\n") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except FileNotFoundError:
            with open(path, "w", newline="\n") as f:
                f.write(line)
                f.flush()

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def __init__(
        self,
        rank: int = 0,
        headless: bool = True,
        frameskip: int = 2,
        screenshot_every: int = 30,
        obs_size: int = 84,
        sprite_slots: int = 12,
        verbose: int = 1,
        reset_mode: str = "soft",
        state_dir: Optional[str] = None,
        ram_timeout_s: float = 10.0,
        reset_timeout_s: float = 60.0,
        seed: Optional[int] = None,
        novelty_enabled: bool = True,
        enable_cell_exploration: bool = False,
        model_name: str = "default",
        action_type: str = "discrete",
        cell_bonus_mode: str = "linear",
        bizhawk_root: Optional[str] = None,
        rom_path: Optional[str] = None,
        exploration_bonus: float = 1.0,
        progress_scale: float = 0.01,
        score_reward: float = 0.01,
        death_penalty: float = -5.0,
        win_bonus: float = 1000.0,
        stuck_penalty: float = 0.0,
        stuck_steps: int = 600,
        novelty_bonus: float = 0.01,
        novelty_max_items: int = 200_000,
        clamp_negative_progress: bool = False,
        cell_bins: tuple[int, int] = (64, 64),
        normalize_progress_by_frameskip: bool = True,
        return_full_res: bool = False,
        keep_screenshots: bool = True,
        fixed_savestate_index: Optional[int] = None,
    ):
        super().__init__()
        atexit.register(self.close)

        self.rank = int(rank)

        # ---- Resolve paths ----
        if bizhawk_root is None:
            bizhawk_root = os.environ.get("BIZHAWK_ROOT")

        if bizhawk_root is None:
            import sys

            if sys.platform.startswith("win"):
                bizhawk_root = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64"
            else:
                bizhawk_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "BizHawk-2.11-linux-x64")
                )
        self.bizhawk_root = bizhawk_root

        if rom_path is None:
            rom_path = os.environ.get("BIZHAWK_ROM")

        if rom_path is None:
            import sys

            if sys.platform.startswith("win"):
                rom_path = os.path.join(self.bizhawk_root, "games", "Roms", "Super Mario World (U) [!].smc")
            else:
                rom_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "games", "Roms", "Super Mario World (U) [!].smc")
                )
        self.rom_path = rom_path

        # ---- seeding ----
        if seed is None:
            seed = 12345 + self.rank
        self._seed = int(seed)
        random.seed(self._seed)
        np.random.seed(self._seed)

        # ---- config ----
        self.headless = bool(headless)
        self._frameskip = int(frameskip)
        self._screenshot_every = int(screenshot_every)
        self._obs_size = int(obs_size)
        self._sprite_slots = int(sprite_slots)
        self._verbose = int(verbose)
        self._reset_mode = str(reset_mode)
        self._ram_timeout_s = float(ram_timeout_s)
        self._reset_timeout_s = float(reset_timeout_s)

        self._exploration_bonus = float(exploration_bonus)
        self._progress_scale = float(progress_scale)
        self._score_reward = float(score_reward)
        self._death_penalty = float(death_penalty)
        self._win_bonus = float(win_bonus)
        self._stuck_penalty = float(stuck_penalty)
        self._stuck_steps = int(stuck_steps)
        self._normalize_progress_by_frameskip = bool(normalize_progress_by_frameskip)
        self._return_full_res = bool(return_full_res)
        self._keep_screenshots = bool(keep_screenshots)
        self._fixed_savestate_index = None if fixed_savestate_index in (None, 0) else int(fixed_savestate_index)

        self._novelty_enabled = bool(novelty_enabled)
        self._novelty_bonus = float(novelty_bonus)
        self._novelty_max_items = int(novelty_max_items)
        self._clamp_negative_progress = bool(clamp_negative_progress)
        self._enable_cell_exploration = bool(enable_cell_exploration)
        self._cell_bonus_mode = str(cell_bonus_mode)

        self._stuck_max_steps = 3000
        self._stuck_counter = 0
        self._consecutive_timeouts = 0
        self._restart_retry_interval_s = 60

        self._no_progress_timeout_s = 30.0
        self._no_progress_start_frame = None

        # ---- spaces ----
        self._action_type = str(action_type)
        if self._action_type == "discrete":
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
            for d in directions:
                base = list(d) + [1, 0, 0, 0]  # hold Y
                actions.append(tuple(base))  # no jump
                base_jump = list(d) + [1, 1, 0, 0]  # B jump
                actions.append(tuple(base_jump))
            self._discrete_actions = actions
            self.action_space = gym.spaces.Discrete(len(self._discrete_actions))
            assert len(self._discrete_actions) == 18
        elif self._action_type == "multibinary":
            self.action_space = gym.spaces.MultiBinary(8)
        elif self._action_type == "box":
            self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown action_type: {self._action_type}")

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, self._obs_size, self._obs_size), dtype=np.uint8
        )

        # ---- IPC files ----
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../games/emulators/bizhawk/RAM/SNES"))
        os.makedirs(base_path, exist_ok=True)
        self._ram_file = os.path.join(base_path, f"smw_ram_{self.rank}.txt")
        self._ctrl_file = os.path.join(base_path, f"smw_ctrl_{self.rank}.txt")
        self._screenshot_file = os.path.join(base_path, f"smw_screen_{self.rank}.bmp")

        default_state_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../games/emulators/bizhawk/SNES/State/")
        )
        self._state_dir = state_dir or default_state_dir

        self._exploration_dir = os.path.join(os.path.dirname(__file__), f"explore_{model_name}_rank{self.rank}")
        os.makedirs(self._exploration_dir, exist_ok=True)

        # exploration tracking
        self._cell_bins = (int(cell_bins[0]), int(cell_bins[1]))

        # caches/trackers
        self._last_obs = np.zeros((1, self._obs_size, self._obs_size), dtype=np.uint8)
        self._last_screenshot_mtime = 0.0
        self._last_full_res = None

        self._last_score = None
        self._last_lives = None
        self._last_level = None
        self._last_x_pos = None
        self._last_y_pos = None

        self._last_emu_frame = -1
        self._frame_stuck_steps = 0

        # cmd id
        self._command_id = 0

        # logs + proc
        self._bizhawk_proc = None
        self._stdout_log = None
        self._stderr_log = None

        # Clean stale IPC/logs for this rank (helps after crashes)
        for p in [
            self._ram_file,
            self._ctrl_file,
            self._screenshot_file,
            self._ram_file + ".lua_log",
            self._ram_file + ".lua_err",
        ]:
            self._safe_unlink(p)

        # Make sure files exist with sane initial content
        # Use cmd_id=-1 so the Lua side ignores it and doesn't overwrite the handshake (-888).
        zeros = ["0"] * (13 + self._sprite_slots * 3)
        self._atomic_write_line(self._ram_file, ",".join(zeros) + "\n")
        self._atomic_write_line(self._ctrl_file, "ACT,0,0,0,0,0,0,0,0,-1\n")

        # Start emulator and wait for handshake
        self._start_bizhawk()

        # Write initial ACT only AFTER handshake
        if self._action_type == "discrete":
            default_action = 0
        elif self._action_type == "multibinary":
            default_action = [0] * 8
        else:
            default_action = [0.0] * 8
        self._write_act(default_action, cmd_id=self._command_id)

        # Rewarder
        self.rewarder = RewardSMW(
            progress_scale=self._progress_scale,
            win_bonus=self._win_bonus,
            death_penalty=self._death_penalty,
            stuck_steps=self._stuck_steps,
            stuck_penalty=self._stuck_penalty,
            use_novelty=self._novelty_enabled,
            novelty_bonus=self._novelty_bonus,
            novelty_max_items=self._novelty_max_items,
            clamp_negative_progress=self._clamp_negative_progress,
            enable_cell_exploration=self._enable_cell_exploration,
            cell_bonus_mode=self._cell_bonus_mode,
            exploration_bonus=self._exploration_bonus,
            cell_bins=self._cell_bins,
        )

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---------------- process cleanup helpers ----------------
    def _kill_rank_processes(self) -> None:
        lua_name = f"smw_rl_control_{self.rank}.lua"
        try:
            out = subprocess.check_output(["bash", "-lc", f"pgrep -fa {shlex.quote(lua_name)} || true"], text=True)
        except Exception:
            return

        pids = []
        for line in out.strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                pids.append(int(parts[0]))
            except Exception:
                continue

        if not pids:
            return

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

        time.sleep(0.3)

        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass

    # ---------------- IPC writers ----------------
    def _write_act(self, action, cmd_id: Optional[int] = None) -> None:
        if cmd_id is None:
            cmd_id = self._command_id

        if self._action_type == "discrete":
            button_vector = self._discrete_actions[int(action)]
        elif self._action_type == "multibinary":
            button_vector = action
        else:  # box
            button_vector = [1 if float(b) >= 0.5 else 0 for b in action]

        row = ["ACT"] + [str(int(b)) for b in button_vector] + [str(int(cmd_id))]
        self._atomic_write_line(self._ctrl_file, ",".join(row) + "\n")

    def _write_load(self, savestate_path: str, cmd_id: Optional[int] = None) -> None:
        if cmd_id is None:
            cmd_id = self._command_id
        p = savestate_path.replace("\\", "/")
        row = ["LOAD", p, str(int(cmd_id))]
        self._atomic_write_line(self._ctrl_file, ",".join(row) + "\n")

    def _read_ram_once(self):
        if not os.path.exists(self._ram_file):
            return None
        try:
            with open(self._ram_file, "r") as f:
                line = f.readline()
            if not line:
                return None
            parts = line.strip().split(",")
            if len(parts) < 13:
                return None
            return [int(p) for p in parts]
        except Exception:
            return None

    def _parse_sprite_slots(self, ram_data):
        tail = ram_data[13:]
        needed = self._sprite_slots * 3
        if len(tail) < needed:
            tail = tail + [0] * (needed - len(tail))
        alive = np.zeros((self._sprite_slots,), dtype=np.float32)
        xy = np.zeros((self._sprite_slots, 2), dtype=np.float32)
        for i in range(self._sprite_slots):
            base = i * 3
            alive[i] = float(tail[base])
            xy[i, 0] = float(tail[base + 1])
            xy[i, 1] = float(tail[base + 2])
        return alive, xy

    def _wait_for_cmd_ack(self, cmd_id: int, timeout_s: float):
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            last = self._read_ram_once()
            if last is not None and int(last[-1]) == int(cmd_id):
                return last
            time.sleep(0.002)
        return None

    # ---------------- Lua script ----------------
    def _create_lua_script(self) -> str:
        lua_dir = os.path.join(self.bizhawk_root, "Lua", "SNES")
        os.makedirs(lua_dir, exist_ok=True)
        target_lua = os.path.join(lua_dir, f"smw_rl_control_{self.rank}.lua")

        lua_ram = self._ram_file.replace("\\", "/")
        lua_ctrl = self._ctrl_file.replace("\\", "/")
        lua_screenshot = self._screenshot_file.replace("\\", "/")
        lua_log = (self._ram_file + ".lua_log").replace("\\", "/")

        lua_content = f'''\
local function log(msg)
    local f = io.open("{lua_log}", "a")
    if f then
        f:write("LUA DEBUG: " .. tostring(msg) .. "\\n")
        f:close()
    end
    io.stdout:write("LUA DEBUG: " .. tostring(msg) .. "\\n")
    io.stdout:flush()
end

local function read_u16_le(addr)
    if memory.read_u16_le then
        return memory.read_u16_le(addr)
    else
        local lo = memory.read_u8(addr)
        local hi = memory.read_u8(addr + 1)
        return lo + 256 * hi
    end
end

log("--- Mario RL Lua Script Starting (Rank {self.rank}) ---")
local ctrl_file = "{lua_ctrl}"
local ram_file = "{lua_ram}"
local screenshot_file = "{lua_screenshot}"

local frames_to_advance = {int(self._frameskip)}
local screenshot_every = {int(self._screenshot_every)}
local last_processed_id = -1
local slots = {int(self._sprite_slots)}
local buttons_map = {{
  "P1 Right", "P1 Left", "P1 Up", "P1 Down",
  "P1 Y", "P1 B", "P1 A", "P1 X"
}}

local heartbeat_every = 60
local last_heartbeat = 0

function write_ram(cmd_id)
    local score = read_u16_le(0x0F34)
    local coins = memory.read_u8(0x0DBF)
    local lives = memory.read_u8(0x0DBE)
    local level = memory.read_u8(0x13BF)
    local x_pos = read_u16_le(0x0094)
    local y_pos = read_u16_le(0x0096)
    local state = memory.read_u8(0x0071)
    local timer = memory.read_u8(0x0F31)*100 + memory.read_u8(0x0F32)*10 + memory.read_u8(0x0F33)
    local game_mode = memory.read_u8(0x0100)
    local frame = emu.framecount()

    local layer1_x = read_u16_le(0x001A)
    local layer1_y = read_u16_le(0x001C)

    local sprite_parts = {{}}
    for i = 0, slots - 1 do
        local status = memory.read_u8(0x14C8 + i)
        local alive = (status ~= 0) and 1 or 0
        local x = memory.read_u8(0x00E4 + i) + 256 * memory.read_u8(0x14E0 + i)
        local y = memory.read_u8(0x00D8 + i) + 256 * memory.read_u8(0x14D4 + i)
        local sx = x - layer1_x
        local sy = y - layer1_y
        table.insert(sprite_parts, tostring(alive))
        table.insert(sprite_parts, tostring(sx))
        table.insert(sprite_parts, tostring(sy))
    end
    local head = string.format("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
        score, coins, lives, level, x_pos, y_pos, layer1_x, layer1_y, state, timer, game_mode, frame, cmd_id)
    local sprite_str = table.concat(sprite_parts, ",")
    local line
    if sprite_str ~= "" then
        line = head .. "," .. sprite_str .. "," .. tostring(cmd_id) .. "\\n"
    else
        line = head .. "," .. tostring(cmd_id) .. "\\n"
    end

    local f = io.open(ram_file, "w")
    if f then
        f:write(line)
        f:flush()
        f:close()
    else
        log("ERROR: Could not open ram_file for writing: " .. ram_file)
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
    if p and p ~= "" then
        local ok, err = pcall(function() savestate.load(p) end)
        if not ok then log("LOAD ERROR: " .. tostring(err)) end
    end
    emu.frameadvance()
    maybe_screenshot(cmd_id, true)
    write_ram(cmd_id)
    return true
  end

  if cmd_type == "SAVE" then
    local p = parts[2]
    if p and p ~= "" then
        local ok, err = pcall(function() savestate.save(p) end)
        if not ok then log("SAVE ERROR: " .. tostring(err)) end
    end
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

local function main_loop()
    local f = io.open(ram_file, "w")
    if f then
        local zeros = {{}}
        for i = 1, slots * 3 do
            table.insert(zeros, "0")
        end
        local head = "0,0,0,0,0,0,0,0,0,0,0,0,-888"
        local line = head .. "," .. table.concat(zeros, ",") .. ",-888\\n"
        f:write(line)
        f:flush()
        f:close()
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
end

local ok, err = pcall(main_loop)
if not ok then
    local err_file = ram_file .. ".lua_err"
    local f = io.open(err_file, "w")
    if f then
        f:write(tostring(err) .. "\\n")
        f:close()
    end
    log("CRITICAL LUA ERROR: " .. tostring(err))
end
'''
        with open(target_lua, "w", newline="\n") as f:
            f.write(lua_content)
        return target_lua

    # ---------------- process management ----------------
    def _start_bizhawk(self) -> None:
        import sys

        is_windows = sys.platform.startswith("win")
        is_linux = sys.platform.startswith("linux")

        if is_windows:
            bizhawk_path = os.path.join(self.bizhawk_root, "games", "emulators", "bizhawk", "EmuHawk.exe")
            lua_script = self._create_lua_script()
            cmd = [bizhawk_path, "--audiosync", "false", "--gdi", "--chromeless", "--lua", lua_script, self.rom_path]
            startupinfo = None
            if self.headless:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            if self._verbose:
                print(f"--- [Rank {self.rank}] Starting BizHawk [Windows] ---")
            self._bizhawk_proc = subprocess.Popen(cmd, startupinfo=startupinfo)
            return

        if not is_linux:
            raise RuntimeError("Unsupported OS for BizHawk launch.")

        # Kill any previous same-rank processes
        self._kill_rank_processes()

        # Nuke config that can pop modal
        cfg = os.path.join(self.bizhawk_root, "config.ini")
        try:
            if os.path.exists(cfg):
                os.replace(cfg, cfg + ".bak")
        except Exception:
            pass

        emu_hawk_exe = os.path.join(self.bizhawk_root, "EmuHawk.exe")
        lua_script = self._create_lua_script()

        if not os.path.isfile(lua_script):
            raise FileNotFoundError(f"Lua script not found: {lua_script}")
        if not os.path.isfile(emu_hawk_exe):
            raise FileNotFoundError(f"EmuHawk.exe not found: {emu_hawk_exe}")
        if not os.path.isfile(self.rom_path):
            raise FileNotFoundError(f"ROM file not found: {self.rom_path}")

        env = os.environ.copy()
        libpath = "/usr/lib/x86_64-linux-gnu"
        env["LD_LIBRARY_PATH"] = f"{self.bizhawk_root}/dll:{self.bizhawk_root}:{libpath}:{env.get('LD_LIBRARY_PATH','')}"
        env["MONO_CRASH_NOFILE"] = "1"
        env["MONO_WINFORMS_XIM_STYLE"] = "disabled"
        env["ALSOFT_DRIVERS"] = "pulse"
        env["PULSE_SINK"] = "DummySink"

        # logs
        log_dir = os.path.join(os.path.dirname(self._ram_file), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._stdout_log = open(os.path.join(log_dir, f"bizhawk_{self.rank}.stdout.log"), "w")
        self._stderr_log = open(os.path.join(log_dir, f"bizhawk_{self.rank}.stderr.log"), "w")

        # Minimal launch command (no xdotool modal handling loop)
        inner_cmd = (
            f"mono {shlex.quote(emu_hawk_exe)} --audiosync false --gdi --chromeless "
            f"--lua {shlex.quote(lua_script)} {shlex.quote(self.rom_path)}"
        )

        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24 -nolisten tcp", "bash", "-lc", inner_cmd]

        if self._verbose:
            print(f"--- [Rank {self.rank}] Starting BizHawk [Linux] ---")
            print(f"    CWD: {self.bizhawk_root}")
            print(f"    CMD: {shlex.join(cmd)}")

        self._bizhawk_proc = subprocess.Popen(
            cmd,
            stdout=self._stdout_log,
            stderr=self._stderr_log,
            cwd=self.bizhawk_root,
            env=env,
            preexec_fn=os.setsid,
        )

        # Wait for handshake (-888)
        if self._verbose:
            print(f"--- [Rank {self.rank}] Waiting for Lua handshake (-888)... ---")

        deadline = time.time() + self._reset_timeout_s
        while time.time() < deadline:
            if self._bizhawk_proc.poll() is not None:
                break
            ram = self._read_ram_once()
            if ram is not None and ram[-1] == -888:
                if self._verbose:
                    print(f"--- [Rank {self.rank}] Lua ready! ---")
                return
            time.sleep(0.2)

        # On failure, check lua_err
        err_msg = ""
        err_file = self._ram_file + ".lua_err"
        if os.path.exists(err_file):
            try:
                with open(err_file, "r") as f:
                    err_msg = f.read().strip()
            except Exception:
                err_msg = ""

        self.close()
        raise RuntimeError(
            f"Rank {self.rank}: Lua handshake timeout after {self._reset_timeout_s}s."
            f"{(' Lua err: ' + err_msg) if err_msg else ''}  "
            f"Check logs in: {log_dir}"
        )

    def close(self) -> None:
        proc = getattr(self, "_bizhawk_proc", None)
        if proc is not None:
            try:
                pgid = os.getpgid(proc.pid)
            except Exception:
                pgid = None

            # graceful
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    proc.terminate()
            except Exception:
                pass

            # wait then hard kill
            try:
                proc.wait(timeout=10)
            except Exception:
                try:
                    if pgid is not None:
                        os.killpg(pgid, signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass

            self._bizhawk_proc = None

        for attr in ("_stdout_log", "_stderr_log"):
            f = getattr(self, attr, None)
            if f:
                try:
                    f.close()
                except Exception:
                    pass
                try:
                    delattr(self, attr)
                except Exception:
                    pass

    # ---------------- observation ----------------
    def _get_obs(self):
        if not os.path.exists(self._screenshot_file):
            return self._last_obs
        try:
            mtime = os.path.getmtime(self._screenshot_file)
            if mtime == self._last_screenshot_mtime:
                return self._last_obs
            full_res = cv2.imread(self._screenshot_file, cv2.IMREAD_COLOR)
            if full_res is None:
                return self._last_obs
            self._last_full_res = full_res
            gray = cv2.cvtColor(full_res, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] != self._obs_size or gray.shape[1] != self._obs_size:
                gray = cv2.resize(gray, (self._obs_size, self._obs_size), interpolation=cv2.INTER_NEAREST)
            self._last_obs = gray.reshape((1, self._obs_size, self._obs_size))
            self._last_screenshot_mtime = mtime
            if not self._keep_screenshots:
                self._safe_unlink(self._screenshot_file)
            return self._last_obs
        except Exception:
            return self._last_obs

    # ---------------- savestates ----------------
    def _list_savestates(self):
        if not os.path.isdir(self._state_dir):
            return []
        states = []
        for f in os.listdir(self._state_dir):
            if f.endswith(".State") and "SMW" in f:
                states.append(os.path.abspath(os.path.join(self._state_dir, f)))
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

    # ---------------- Gym API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._reset_mode == "hard":
            self.close()
            self._start_bizhawk()

        # per-episode exploration reset
        self._stuck_counter = 0

        states = self._list_savestates()
        if not states:
            raise RuntimeError(f"No savestates found in {self._state_dir}")

        if options and options.get("savestate"):
            chosen = options["savestate"]
        elif self._fixed_savestate_index is not None:
            chosen = self.get_savestate_by_index(self._fixed_savestate_index)
        else:
            chosen = random.choice(states)

        self._command_id += 1
        self._write_load(chosen, cmd_id=self._command_id)

        ram_data = self._wait_for_cmd_ack(self._command_id, timeout_s=self._reset_timeout_s)
        if ram_data is None:
            raise RuntimeError(f"Rank {self.rank}: reset LOAD timeout (cmd_id={self._command_id})")

        score, coins, lives, level, x_pos, y_pos, layer1_x, layer1_y, state, timer, game_mode, frame, cmd_id = ram_data[:13]
        sprite_alive, sprite_xy = self._parse_sprite_slots(ram_data)

        self._last_score = score
        self._last_lives = lives
        self._last_level = level
        self._last_x_pos = x_pos
        self._last_y_pos = y_pos
        self._last_emu_frame = frame
        self._frame_stuck_steps = 0
        self._no_progress_start_frame = frame

        self.rewarder.reset()

        obs = self._get_obs()
        info = {
            "savestate": chosen,
            "level": level,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "layer1_x": layer1_x,
            "layer1_y": layer1_y,
            "mario_screen_x": x_pos - layer1_x,
            "mario_screen_y": y_pos - layer1_y,
            "cmd_id": cmd_id,
            "sprite_alive": sprite_alive,
            "sprite_xy": sprite_xy,
        }
        if self._return_full_res:
            info["full_res_frame"] = self._last_full_res
        return obs, info

    def step(self, action):
        self._command_id += 1
        self._write_act(action, cmd_id=self._command_id)

        ram_data = self._wait_for_cmd_ack(self._command_id, timeout_s=self._ram_timeout_s)
        if ram_data is None:
            self._consecutive_timeouts += 1
            if self._verbose:
                print(
                    f"[Rank {self.rank}] RAM timeout at cmd_id={self._command_id} "
                    f"({self._consecutive_timeouts} in a row)"
                )

            if self._verbose:
                print(
                    f"[Rank {self.rank}] Restarting BizHawk after timeout; "
                    f"retrying every {self._restart_retry_interval_s}s until it comes back."
                )
            self._restart_bizhawk_with_retry(reason="ram_timeout")
            self._consecutive_timeouts = 0

            obs = self._get_obs()
            return obs, 0.0, False, False, {"error": "ram_timeout", "restarted": True}

        self._consecutive_timeouts = 0

        obs = self._get_obs()
        score, coins, lives, level, x_pos, y_pos, layer1_x, layer1_y, state, timer, game_mode, frame, cmd_id = ram_data[:13]
        sprite_alive, sprite_xy = self._parse_sprite_slots(ram_data)

        if self._verbose >= 2:
            active = np.where(sprite_alive > 0.5)[0].tolist()
            if active:
                coords = [(int(i), float(sprite_xy[i, 0]), float(sprite_xy[i, 1])) for i in active]
                print(f"[Rank {self.rank}] Sprites: {coords}")

        # frame stuck detection
        if frame == self._last_emu_frame:
            self._frame_stuck_steps += 1
        else:
            self._frame_stuck_steps = 0
            self._last_emu_frame = frame

        if self._frame_stuck_steps > (self._frameskip * 5):
            if self._verbose:
                print(f"[Rank {self.rank}] frame stuck, restarting BizHawk")
            self._restart_bizhawk_with_retry(reason="frame_stuck")
            obs = self._get_obs()
            return obs, 0.0, False, False, {"error": "frame_stuck_restart", "restarted": True}

        # no progress timeout
        if self._no_progress_start_frame is None:
            self._no_progress_start_frame = frame
        if self._last_x_pos is not None and self._last_y_pos is not None:
            if x_pos != self._last_x_pos or y_pos != self._last_y_pos:
                self._no_progress_start_frame = frame

        elapsed_frames = frame - self._no_progress_start_frame
        if elapsed_frames >= int(self._no_progress_timeout_s * 60):
            if self._verbose:
                print(f"[Rank {self.rank}] No progress for {self._no_progress_timeout_s}s, truncating")
            return obs, 0.0, False, True, {"error": "no_progress_timeout"}

        # termination logic
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

        terminated = died or won
        truncated = False
        if self._stuck_counter >= self._stuck_max_steps:
            truncated = True
            if self._verbose:
                print(f"[Rank {self.rank}] Truncated (stuck)")

        # Calculate Reward via RewardSMW
        # Calculate Reward via RewardSMW
        # 1. Base rewards (Progress, Novelty, Win/Loss, Stuck) + Cell Exploration
        ram_dict = {"x_pos": x_pos, "y_pos": y_pos}
        reward, reward_info = self.rewarder.step_reward(
            obs_img=obs,
            ram=ram_dict,
            terminated=terminated,
            truncated=truncated,
            info={"win": won, "death": died}
        )

        # Env-level stuck counter for truncation
        # Use RewardSMW info to determine if we are "stuck" relative to its metrics
        # If RewardSMW says we made progress or found novelty, reset counter.
        # Also check local progress (x_pos > last_x_pos) as a looser stuck check?
        # User requested to remove redundancy.
        # Let's rely on what RewardSMW considers "new" + local progress check for loose movement.
        
        found_something = reward_info.get("new_progress", False) or reward_info.get("new_novelty", False) or reward_info.get("new_cell", False)
        
        # Keep local progress check for truncation safety? 
        # If agent is moving forward (even if not max progress), we probably shouldn't truncate.
        if x_pos > (self._last_x_pos if self._last_x_pos is not None else -99999):
             found_something = True

        if found_something:
            self._stuck_counter = 0
        else:
            self._stuck_counter += 1

        # Env-level stuck counter for truncation (RewardSMW has its own penalty logic but doesn't truncate)
        # We need to manually update stuck counter for truncation purposes if we are relying on novelty/progress
        # Wait, RewardSMW tracks `steps_since_progress`. We can query it?
        # Or just keep the existing `stuck_counter` logic relative to Novelty/Exploration?
        
        # Original logic: 
        # if novelty/cell found: stuck_counter = 0
        # else: stuck_counter += 1
        
        # RewardSMW doesn't expose "did I find novelty?".
        # BUT we can check if reward > 0? No, progress gives reward too.
        # Let's keep `self._stuck_counter` logic for TRUNCATION separate from RewardSMW's penalty.
        # Although this is slightly redundant.

        # update trackers
        self._last_score = score
        self._last_lives = lives
        self._last_level = level
        self._last_x_pos = x_pos
        self._last_y_pos = y_pos

        if terminated or truncated:
            self.rewarder.reset() # RewardSMW has its own seen_hashes and cell exploration

        info = {
            "score": score,
            "coins": coins,
            "lives": lives,
            "level": level,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "layer1_x": layer1_x,
            "layer1_y": layer1_y,
            "mario_screen_x": x_pos - layer1_x,
            "mario_screen_y": y_pos - layer1_y,
            "timer": timer,
            "state": state,
            "game_mode": game_mode,
            "frame": frame,
            "cmd_id": cmd_id,
            "win": won,
            "death": died,
            "timeout_death": timeout_death,
            "sprite_alive": sprite_alive,
            "sprite_xy": sprite_xy,
            "reward_progress": reward_info.get("reward_progress", 0.0),
            "reward_novelty": reward_info.get("reward_novelty", 0.0),
            "reward_cell": reward_info.get("reward_cell", 0.0),
            "reward_win": reward_info.get("reward_win", 0.0),
            "reward_death": reward_info.get("reward_death", 0.0),
            "reward_stuck": reward_info.get("reward_stuck", 0.0),
        }
        if self._return_full_res:
            info["full_res_frame"] = self._last_full_res
        return obs, float(reward), bool(terminated), bool(truncated), info

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


# ---------------- Standalone runner ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--bizhawk_root", type=str, default="/home/simon/snesRL/BizHawk-2.11-linux-x64")
    parser.add_argument("--rom_path", type=str, default=None)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--screenshot_every", type=int, default=30)
    args = parser.parse_args()

    env = MarioBizHawkEnv(
        bizhawk_root=args.bizhawk_root,
        rom_path=args.rom_path,
        headless=bool(args.headless),
        verbose=int(args.verbose),
        rank=0,
        screenshot_every=int(args.screenshot_every),
    )

    def _cleanup(*_):
        try:
            env.close()
        except Exception:
            pass

    atexit.register(_cleanup)

    def _sig_handler(signum, frame):
        _cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    if args.level == 0:
        obs, info = env.reset()
    else:
        savestate = env.get_savestate_by_index(args.level)
        obs, info = env.reset(options={"savestate": savestate})

    print(f"Loaded level: {info['level']}, savestate: {info['savestate']}")

    """
        # Construct the internal bash command to run inside xvfb-run
        # 1. Start PulseAudio with a null sink (Restored for robustness)
        # 2. Launch BizHawk
        # 3. Targeted background loop to dismiss modals if they appear
        # inner_cmd = (
        #     "pulseaudio --start --exit-idle-time=-1; "
        #     "pactl load-module module-null-sink sink_name=DummySink sink_properties=device.description=DummySink >/dev/null 2>&1 || true; "
        #     "export PULSE_SINK=DummySink; "
        #     f"mono {shlex.quote(emu_hawk_exe)} --audiosync false --gdi --chromeless --lua {shlex.quote(lua_script)} {shlex.quote(self.rom_path)} & "
        #     "BIZ_PID=$!; "
        #     "if ! command -v xdotool >/dev/null 2>&1; then echo \"[Rank %s] xdotool not found, cannot dismiss dialogs\" >&2; fi; " % self.rank +
        #     "for i in {1..60}; do "
        #     "  sleep 1; "
        #     "  if ! kill -0 $BIZ_PID 2>/dev/null; then break; fi; "
        #     "  if command -v xdotool >/dev/null 2>&1; then "
        #     "    xdotool search --name \"Mismatched version in config file\" windowactivate --sync key Return 2>/dev/null; "
        #     "    xdotool search --name \"Couldn't initialize sound device\" windowactivate --sync key Return 2>/dev/null; "
        #     "    xdotool search --name \"Welcome\" windowactivate --sync key Return 2>/dev/null; "
        #     "  fi; "
        #     "done; "
        #     "wait $BIZ_PID"
        # )
        # inner_cmd = (
        #     "pulseaudio --start --exit-idle-time=-1; "
        #     "pactl load-module module-null-sink sink_name=DummySink >/dev/null 2>&1 || true; "
        #     "export PULSE_SINK=DummySink; "
        #     f"mono {shlex.quote(emu_hawk_exe)} "
        #     f"--audiosync false --gdi --chromeless "
        #     f"--lua {shlex.quote(lua_script)} {shlex.quote(self.rom_path)}"
        # )
        inner_cmd = (
            "pulseaudio --start --exit-idle-time=-1; "
            "pactl load-module module-null-sink sink_name=DummySink >/dev/null 2>&1 || true; "
            "export PULSE_SINK=DummySink; "
            f"mono {shlex.quote(emu_hawk_exe)} "
            f"--audiosync false --gdi --chromeless "
            f"--lua {shlex.quote(lua_script)} {shlex.quote(self.rom_path)} & "
            "BIZ_PID=$!; "
            "sleep 3; "
            "xdotool search --onlyvisible --name '.*' key Return 2>/dev/null || true; "
            "wait $BIZ_PID"
        )

        inner_cmd = (
            f"mono {shlex.quote(emu_hawk_exe)} "
            f"--audiosync false --gdi --chromeless "
            f"--lua {shlex.quote(lua_script)} {shlex.quote(self.rom_path)}"
        )

    """
