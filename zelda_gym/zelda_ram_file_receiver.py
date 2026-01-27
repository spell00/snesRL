import time
import os

def read_ram_file(path):
    while True:
        if os.path.exists(path):
            with open(path, "r") as f:
                line = f.readline()
                if line:
                    try:
                        rupees, hearts, max_hearts, room_id = map(int, line.strip().split(','))
                        print(f"Rupees: {rupees}, Hearts: {hearts}/8, Max Hearts: {max_hearts}/8, Room ID: {room_id}")
                    except Exception as e:
                        print("Error parsing line:", line, e)
        time.sleep(0.1)

if __name__ == "__main__":
    read_ram_file("zelda_ram.txt")
