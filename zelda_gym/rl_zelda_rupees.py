from zelda_snes_env import ZeldaSnesEnv
import time
import numpy as np
import cv2

# Update these paths as needed
SNES9X_PATH = 'snes9x-x64.exe'  # Path to your Snes9x executable
ROM_PATH = 'Roms/Legend of Zelda, The - A Link to the Past (USA).sfc'  # Path to your ROM

# Define a simple policy: move in random directions
ACTIONS = [0, 1, 2, 3]  # up, down, left, right

# Helper to extract rupee count from screen (placeholder)
def extract_rupees(obs):
    # TODO: Implement image recognition or memory reading to get rupee count
    # For now, return a random value as a placeholder
    return np.random.randint(0, 100)

env = ZeldaSnesEnv(rom_path=ROM_PATH, snes9x_path=SNES9X_PATH)

# Reset environment and just display the emulator screen, do nothing else
obs, info = env.reset()
print("Zelda loaded. Press 'q' in the window to exit.")
while True:
    obs = env._get_obs()
    cv2.imshow('Zelda Emulator', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.05)

env.close()
cv2.destroyAllWindows()
