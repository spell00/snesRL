from zelda_snes_env import ZeldaSnesEnv
import time

# Update these paths as needed
SNES9X_PATH = '../snes9x.exe'  # Path to your Snes9x executable
ROM_PATH = '../Roms/Legend of Zelda, The - A Link to the Past (USA).sfc'  # Path to your ROM

env = ZeldaSnesEnv(rom_path=ROM_PATH, snes9x_path=SNES9X_PATH)

obs, info = env.reset()
for i in range(20):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    print(f'Step {i}, Reward: {reward}, Done: {done}')
    time.sleep(0.2)
    if done:
        break

env.close()
