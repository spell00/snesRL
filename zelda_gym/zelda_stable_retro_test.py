import retro
import cv2
import time

# Make sure your ROM is imported into stable-retro's system (see docs for importing)
# This example assumes the game is named 'LegendOfZelda-A Link to the Past-Snes'
# You may need to run: python -m retro.import ROM_PATH

env = retro.make(game='LegendOfZelda-A Link to the Past-Snes')
obs = env.reset()
print("Zelda loaded via stable-retro. Press 'q' in the window to exit.")

while True:
    env.render()  # This will open the emulator window
    # Show the observation in a separate OpenCV window (optional)
    cv2.imshow('Zelda Emulator (stable-retro)', obs[:, :, ::-1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Take no action (idle)
    obs, _, done, _ = env.step([0] * env.action_space.shape[0])
    if done:
        obs = env.reset()
    time.sleep(0.05)

env.close()
cv2.destroyAllWindows()
