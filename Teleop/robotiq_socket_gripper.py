"""Class using socket to control a Robotiq 2F-140 gripper"""
"""Allows for connection, activation and movement of gripper; also enables setting speed and force of gripper"""
"""Pose, speed and force are defined from 0-255"""

import socket, time

class RobotiqSocketGripper:
    def __init__(self, host, port=63352, timeout=2):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def send_command(self, cmd: str) -> str:
        full = (cmd.strip() + "\n").encode('utf-8')
        self.sock.sendall(full)
        return self.sock.recv(1024).decode('utf-8').strip()

    def activate(self, speed=255, force=150):
        self.send_command("SET ACT 1")
        time.sleep(0.1)
        self.send_command("SET GTO 1")
        self.send_command(f"SET SPE {speed}")
        self.send_command(f"SET FOR {force}")
        time.sleep(0.5)

    def set_speed(self, speed:int):
        speed = max(0, min(255, speed))
        return self.send_command(f"SET SPE {speed}")

    def set_force(self, force:int):
        force = max(0, min(255, force))
        return self.send_command(f"SET FOR {force}")

    def move(self, pos:int):
        pos = max(0, min(255, pos))
        return self.send_command(f"SET POS {pos}")

    def get_pos(self) -> int:
        resp = self.send_command("GET POS")
        return int(resp.split()[-1]) if resp.startswith("POS") else None

    def get_speed(self) -> int:
        resp = self.send_command("GET SPE")
        return int(resp.split()[-1]) if resp.startswith("SPE") else None

    def get_force(self) -> int:
        resp = self.send_command("GET FOR")
        return int(resp.split()[-1]) if resp.startswith("FOR") else None

    def get_obj(self) -> int:
        resp = self.send_command("GET OBJ")
        return int(resp.split()[-1]) if resp.startswith("OBJ") else None

    def get_status(self) -> dict:
        """Fetch pos, speed, force, and object-detect flags."""
        return {
            'pos':  self.get_pos(),
            'speed': self.get_speed(),
            'force': self.get_force(),
            'obj':  self.get_obj(),
        }
