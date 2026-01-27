# Zelda Gym Environment

This project provides a reinforcement learning environment for The Legend of Zelda: A Link to the Past, using the SNES9x emulator and the `stable-retro` library.

## Requirements

- **WSL2 (Windows Subsystem for Linux 2)**: This project is designed to run under WSL2 for improved compatibility and performance with Linux-based tools and libraries.
- **stable-retro**: We use the `stable-retro` library, a maintained fork of OpenAI's `retro`, for interfacing with SNES games and creating custom RL environments.
- **Python 3.12+** (recommended)

## Setup

1. **Install WSL2**
   - Follow the official Microsoft guide: https://docs.microsoft.com/en-us/windows/wsl/install
   - Make sure you have a working Linux distribution (e.g., Ubuntu) installed via WSL2.

2. **Install Python and dependencies**
   - Set up a Python virtual environment (recommended):
     ```bash
     python3 -m venv retroenv
     source retroenv/bin/activate
     ```
   - Install requirements:
     ```bash
     pip install -r zelda_gym/requirements.txt
     ```

3. **Install stable-retro**
   - If not included in requirements.txt, install via pip:
     ```bash
     pip install stable-retro
     ```

4. **ROMs and Saves**
   - Place your SNES ROMs in the `Roms/` directory.
   - Save files go in the `Saves/` directory.

## Usage

- Main environment code is in `zelda_gym/zelda_snes_env.py`.
- Example usage and tests are in `zelda_gym/test_zelda_env.py` and `zelda_gym/zelda_stable_retro_test.py`.

## Notes

- This project is intended for research and educational purposes only. Please ensure you own any ROMs you use.
- For best results, always run your Python scripts from within your WSL2 environment.

## References
- [stable-retro GitHub](https://github.com/StableRetros/stable-retro)
- [WSL2 Documentation](https://docs.microsoft.com/en-us/windows/wsl/)
