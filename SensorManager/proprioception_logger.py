"""Function to log proprioception data in JSON format from a UR cobot"""
"""Uses function from get_unique_filename.py"""
"""Tested using a UR3e cobot"""

import json
import time
from rtde_receive import RTDEReceiveInterface
from Teleop.robotiq_socket_gripper import RobotiqSocketGripper

from get_unique_filename import get_unique_filename


def proprioception_logger(root_dir: str, start_event, stop_event,  robot_ip: str):
    """
    Args:
        root_dir: Root directory to save data
        start_event: threading.Event to start logging from
        stop_event: threading.Event to stop logging from
        robot_ip: Static IP address of robot

    Returns:
        A JSON file containing the following proprioception data:
            Elapsed time of data recorded
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
    start_time = None # Initialize timer

    # Capture data
    while not stop_event.is_set():
        # Start recording when start_event is triggered
        if start_event.is_set():
            if start_time is None: # Print initialization status
                start_time = time.time()
                print("Proprioception logger started.")

            # Log during duration
            timestamp = time.time()
            elapsed = timestamp - start_time

            # TCP pose
            tcp_pose_m = rtde_receive.getActualTCPPose()
            tcp_pose_mm = [x * 1000 if i < 3 else x for i, x in enumerate(tcp_pose_m)] # Convert to mm

            # Joint positions
            joint_pos = rtde_receive.getActualQ()

            # Gripper pose
            gripper_pos = gripper.get_pos() or 0 # Absolute (0-255)
            gripper_mm = gripper_pos_to_mm(gripper_pos) # In mm

            # Save log
            log.append({
                "elapsed_time": elapsed,
                "tcp_pose_mm": tcp_pose_mm,
                "joint_positions_rad": joint_pos,
                "gripper_position_0_255": gripper_pos,
                "gripper_position_mm": gripper_mm
            })

        time.sleep(0.1) # 10 Hz frequency of recording

    # Save proprioception data to JSON file
    if log:
        # Create unique filename to prevent overwriting
        file_path = get_unique_filename("proprioception_log", ".json", root_dir)

        # Save file
        with open(file_path, "w") as f:
            json.dump(log, f, indent=4)
        print(f"Proprioception data saved to {file_path}")