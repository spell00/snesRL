import subprocess
import os
import time

# Paths
bizhawk_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\EmuHawk.exe"
rom_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\Roms\Super Mario World (U) [!].smc"
lua_script_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\Lua\SNES\mario_simple_right.lua"
savestate_path = r"C:\Users\simon\Documents\snes9x-1.60-win32-x64\games\emulators\bizhawk\SNES\State\Super Mario World (USA).Snes9x.QuickSave1.State"

# Launch BizHawk with ROM, Lua script, and savestate
cmd = [bizhawk_path, rom_path, "--lua", lua_script_path, "--load-state", savestate_path]
print("Launching BizHawk:", ' '.join(cmd))
proc = subprocess.Popen(cmd)

try:
    time.sleep(20)  # Let Mario move right for 20 seconds
finally:
    proc.terminate()
    print("BizHawk terminated.")
