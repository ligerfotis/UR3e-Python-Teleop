"""To replay UR movements with a Robotiq 2F-140 gripper based on available JSON data, either Cartesian TCP positions or joint angles"""
"""Intended to be used in conjunction with proprioception_logger.py"""
"""Tested with a UR3e cobot"""
"""Activate gripper prior to start"""

"""
Requirements for JSON data:
    TCP Pose: [x, y, z, Rx, Ry, Rz] with respect to base
        OR
    Joint angles: Angles of all 6 joints starting from the base
    Robotiq gripper: Finger distance (0-255)
      
Units:
    Distance => mm
    Angles => radians
    Time => seconds
    Velocity => m/s
    Acceleration => m/s^2
"""

import json
import time
from rtde_control import RTDEControlInterface
from Teleop.robotiq_socket_gripper import RobotiqSocketGripper

# Set parameters
robot_ip = "192.168.1.223"
json_file = "proprioception_log.json"
frequency = 100.0  # Frequency of robot control (Hz)
use_tcp_pose = True  # Set to True if you want to replay using TCP pose instead of joint angles

time_step = 1.0 / frequency

# Connect to robot and gripper
rtde_control = RTDEControlInterface(robot_ip)
gripper = RobotiqSocketGripper(robot_ip)
gripper.connect()

# Load JSON file
with open(json_file, "r") as f:
    log = json.load(f)
print(f"Starting replay ({len(log)} movements).")

# Activate servo mode
rtde_control.servoStop() # Ensure servo is off
time.sleep(0.1)

# Move to initial position
if use_tcp_pose:
    print("Moving to initial TCP pose.")
    tcp_pose_mm = log[0]["tcp_pose_mm"]
    # Convert from mm to m
    tcp_pose_m = [
        tcp_pose_mm[0] / 1000.0,
        tcp_pose_mm[1] / 1000.0,
        tcp_pose_mm[2] / 1000.0,
        tcp_pose_mm[3],
        tcp_pose_mm[4],
        tcp_pose_mm[5]
    ]
    rtde_control.moveL(
        tcp_pose_m, # Initial position
        0.1, # Velocity
        0.1 # Acceleration
    )
else:
    print("Moving to initial joint position.")
    rtde_control.moveJ(
        log[0]["joint_positions_rad"], # Initial position
        0.1, # Velocity
        0.2  # Acceleration
    )

# Move gripper to initial position
gripper.move(log[0]["gripper_position_0_255"])
print("Initial pose complete.")

# Replay movements
start_time = time.time()
print("Replaying movements.")
for entry in log:
    elapsed_target = entry["elapsed_time"]
    elapsed_current = time.time() - start_time

    # Synchronize timing
    wait = elapsed_target - elapsed_current
    if wait > 0:
        time.sleep(wait)

    if use_tcp_pose:
        # Convert from mm to m
        tcp_pose_mm = entry["tcp_pose_mm"]
        tcp_pose_m = [
            tcp_pose_mm[0] / 1000.0,
            tcp_pose_mm[1] / 1000.0,
            tcp_pose_mm[2] / 1000.0,
            tcp_pose_mm[3],
            tcp_pose_mm[4],
            tcp_pose_mm[5]
        ]

        # Pass TCP pose to robot
        rtde_control.servoL(
            tcp_pose_m, # TCP Pose
            0.3, # Velocity
            0.3, # Acceleration
            time_step,
            0.1, # Lookahead time
            150 # Gain
        )
    else:
        rtde_control.servoJ(
            entry["joint_positions_rad"], # Joint position
            0.3, # Velocity
            0.3, # Acceleration
            time_step,
            0.1, # Lookahead time
            150 # Gain
        )

    # Move gripper
    gripper.move(entry["gripper_position_0_255"])

# Stop control
rtde_control.servoStop()
rtde_control.stopScript()
print("Replay complete.")