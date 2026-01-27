import socket

HOST = '127.0.0.1'
PORT = 12345

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        data = s.recv(1024)
        if not data:
            break
        # Parse CSV: rupees,hearts,max_hearts,room_id\n
        try:
            rupees, hearts, max_hearts, room_id = map(int, data.decode().strip().split(','))
            print(f"Rupees: {rupees}, Hearts: {hearts}/8, Max Hearts: {max_hearts}/8, Room ID: {room_id}")
        except Exception as e:
            print("Error parsing data:", e)
