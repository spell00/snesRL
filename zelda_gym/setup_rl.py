
import subprocess
import sys
import psutil

def install_packages():
    print("Installing packages: torch, stable-baselines3, shimmy, gymnasium...")
    packages = ["torch", "stable-baselines3", "shimmy", "gymnasium"]
    for package in packages:
        try:
             subprocess.check_call([sys.executable, "-m", "pip", "install", package])
             print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
             print(f"Failed to install {package}")

def check_gpu():
     print("\nChecking GPU availability...")
     try:
         import torch
         if torch.cuda.is_available():
             print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
         else:
             print("CUDA is NOT available. Training will use CPU.")
     except ImportError:
         print("PyTorch not installed, cannot check GPU.")

def check_ram_disk_space():
    print("\nSystem Resources:")
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
    
    # Check current directory disk usage
    disk = psutil.disk_usage('.')
    print(f"Disk Free Space: {disk.free / (1024**3):.2f} GB")

if __name__ == "__main__":
    install_packages()
    check_gpu()
    check_ram_disk_space()
    print("\nSetup check complete. You are ready to create the training script.")
