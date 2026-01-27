import subprocess
import os
import time

# Paths
bizhawk_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\EmuHawk.exe"
rom_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\Roms\Super Mario World (U) [!].smc"
lua_script_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\Lua\SNES\mario_go_right.lua"
action_file = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\RAM\SNES\smw_action_test.txt"

# Write simple Lua script to always press Right
lua_code = '''
local action_file = "' + action_file.replace('\\', '/') + '"
local ram_file = nil -- Not used here

function get_input_for_action()
    local pad = {}
    pad["P1 Right"] = true
    return pad
end

while true do
    joypad.set(get_input_for_action())
    emu.frameadvance()
end
'''

with open(lua_script_path, "w") as f:
    f.write(lua_code)

# Launch BizHawk with ROM and Lua script
cmd = [bizhawk_path, rom_path, "--lua", lua_script_path]
print("Launching BizHawk:", ' '.join(cmd))
proc = subprocess.Popen(cmd)

# Wait for a while to let Mario move right
