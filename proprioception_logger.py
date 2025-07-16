"""Function to log proprioception data in JSON format from UR cobot"""
"""Tested using UR3e"""

import json
import os
import time
from rtde_receive import RTDEReceiveInterface
from robotiq_socket_gripper import RobotiqSocketGripper


def proprioception_logger(root_dir: str, duration: float,  robot_ip: str, sampling_rate: float = 10):
    """
    Args:
        root_dir: Main directory to save data
        duration: Duration in seconds to log proprioception data
        robot_ip: IP address of robot
        sampling_rate: Sampling rate of data in Hz (Default: 10.0 Hz)

    Returns:
        A JSON file containing the following proprioception data:
            Timestamp of data recording
            TCP Pose with respect to base in Cartesian coordinate system [x, y, z, Rx, Ry, Rz]
            Joint angles in radians for 6 joints starting from the base
            Robotiq gripper finger distance

    All linear distances are captured in mm.
    All angles are captured in radians.
    """

    # Connect to robot and gripper
    rtde_receive = RTDEReceiveInterface(robot_ip)
    gripper = RobotiqSocketGripper(robot_ip)
    gripper.connect()

    # Function to convert 2F-140 Robotiq gripper value (0-255) into mm
    def gripper_pos_to_mm(pos_0_255):
        return 140.0 * (1 - (pos_0_255 / 255.0))

    # Proprioception log
    log = []  # Initialize logger
    start_time = time.time() # Initialize timer
    print(f"Proprioception logger started.")

    # Capture proprioception data within duration
    while time.time() - start_time < duration:
        timestamp = time.time()
        elapsed_time = timestamp - start_time

        tcp_pose_m = rtde_receive.getActualTCPPose() # TCP Pose => [x, y, z, Rx, Ry, Rz]
        tcp_pose_mm = [
            tcp_pose_m[0] * 1000, # x in mm
            tcp_pose_m[1] * 1000, # y in mm
            tcp_pose_m[2] * 1000, # z in mm
            tcp_pose_m[3], # Rx in rad
            tcp_pose_m[4], # Ry in rad
            tcp_pose_m[5] # Rz in rad
        ]
        joint_pos = rtde_receive.getActualQ() # Joint angles => [Base, Shoulder, Elbow, Wrist 1, Wrist 2, Wrist 3]
        gripper_pos = gripper.get_pos() or 0 # Gripper finger distance from 0 (open) - 255 (closed)
        gripper_mm = gripper_pos_to_mm(gripper_pos) # Gripper finger distance in mm

        # Store data as a dictionary
        entry = {
            "elapsed_time": elapsed_time,
            "tcp_pose_mm": tcp_pose_mm,
            "joint_positions_rad": joint_pos,
            "gripper_position_0_255": gripper_pos,
            "gripper_position_mm": gripper_mm
        }

        log.append(entry)
        sampling_interval = 1 / sampling_rate
        time.sleep(sampling_interval)

    # Output file name
    file_name = os.path.join(root_dir, "proprioception_log.json")

    # Save proprioception data
    with open(file_name, "w") as f:
        json.dump(log, f, indent=4)

    print(f"Proprioception logging complete. Saved to {file_name}")
