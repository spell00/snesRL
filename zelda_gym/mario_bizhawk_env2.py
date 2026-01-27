import numpy as np
import subprocess
import os
import time
import random

class MarioBizHawkEnv2:
    """
    Simple BizHawk Mario environment (blind, no image, RAM state only).
    Actions: 0=No-op, 1=Right, 2=Left, 3=Jump
    State: [score, coins, lives, x_pos, state, timer]
    """
    def __init__(self, rank=0):
        self.rank = rank
        self.action_space = 4
        self.state_space = 6
        base_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\RAM\SNES"
        self._ram_file = os.path.join(base_path, f"smw_ram2_{self.rank}.txt")
        self._action_file = os.path.join(base_path, f"smw_action2_{self.rank}.txt")
        os.makedirs(base_path, exist_ok=True)
        self._bizhawk_proc = None
        self._savestate = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\SNES\State\Super Mario World (USA).Snes9x.QuickSave1.State"
        self._start_bizhawk()

    def _create_lua_script(self):
        target_lua = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\Lua\SNES\smw_rl_control2_{}.lua".format(self.rank)
        lua_ram = self._ram_file.replace("\\", "/")
        lua_action = self._action_file.replace("\\", "/")
        lua_savestate = self._savestate.replace("\\", "/")
        lua_content = f'''
local action_file = "{lua_action}"
local ram_file = "{lua_ram}"
local savestate_path = "{lua_savestate}"
local first = true
function get_input_for_action(action)
    local pad = {{}}
    if action == 1 then pad["Right"] = true end
    if action == 2 then pad["Left"] = true end
    if action == 3 then pad["A"] = true end
    return pad
end
function read_command()
    local f = io.open(action_file, "r")
    if f then
        local line = f:read("*l")
        f:close()
        local act = tonumber(line)
        return act or 0
    end
    return 0
end
function write_ram()
    local score = memory.read_u16_le(0x0F34)
    local coins = memory.read_u8(0x0DBF)
    local lives = memory.read_u8(0x0DBE)
    local x_pos = memory.read_u16_le(0x0094)
    local state = memory.read_u8(0x0071)
    local t100 = memory.read_u8(0x0F31)
    local t10 = memory.read_u8(0x0F32)
    local t1 = memory.read_u8(0x0F33)
    local timer = t100 * 100 + t10 * 10 + t1
    local f = io.open(ram_file, "w")
    if f then
        f:write(string.format("%d,%d,%d,%d,%d,%d\n", score, coins, lives, x_pos, state, timer))
        f:close()
    end
end
while true do
    if first and savestate_path ~= "" then
        savestate.load(savestate_path)
        first = false
    end
    local action = read_command()
    joypad.set(get_input_for_action(action))
    emu.frameadvance()
    write_ram()
end
'''
        with open(target_lua, "w") as f:
            f.write(lua_content)
        return target_lua

    def _start_bizhawk(self):
        bizhawk_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\EmuHawk.exe"
        rom_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\Roms\Super Mario World (U) [!].smc"
        lua_script = self._create_lua_script()
        args = [bizhawk_path, rom_path, "--lua", lua_script, "--load-state", self._savestate]
        self._bizhawk_proc = subprocess.Popen(args)
        time.sleep(3.0)
        with open(self._action_file, "w") as f:
            f.write("0\n")

    def reset(self):
        with open(self._action_file, "w") as f:
            f.write("0\n")
        time.sleep(0.2)
        return self._read_ram()

    def step(self, action):
        with open(self._action_file, "w") as f:
            f.write(f"{action}\n")
        time.sleep(0.05)
        state = self._read_ram()
        reward = 1.0  # Simple reward for each step
        done = False  # No terminal logic for now
        info = {}
        return state, reward, done, info

    def _read_ram(self):
        for _ in range(100):
            if os.path.exists(self._ram_file):
                with open(self._ram_file, "r") as f:
                    line = f.readline()
                    if line:
                        parts = line.strip().split(',')
                        if len(parts) == 6:
                            return np.array(list(map(int, parts)), dtype=np.int32)
            time.sleep(0.01)
        return np.zeros(self.state_space, dtype=np.int32)

    def close(self):
        if self._bizhawk_proc is not None:
            self._bizhawk_proc.terminate()
            self._bizhawk_proc = None
