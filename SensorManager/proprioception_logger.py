"""Function to log proprioception data in JSON format from a UR cobot based on keyboard input"""
"""Uses function from get_unique_filename.py"""
"""Tested using a UR3e cobot"""
"""Intended to be used with sensor_manager.py and associated functions"""

import json
import time
from rtde_receive import RTDEReceiveInterface
from Teleop.robotiq_socket_gripper import RobotiqSocketGripper

from get_unique_filename import get_unique_filename


def proprioception_logger(root_dir: str, recording_event, stop_event, robot_ip: str):
    """
    Records multiple logs in one program session based on keyboard inputs
    
    Args:
        root_dir: Root directory to save data
        recording_event: threading.Event to start logging from
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

    # Initialize variables
    log = []
    recording = False # To check whether recording is in progress
    start_time = None

    # Main loop
    try:
        print("- Proprioception thread ready.")

        # Initialize variables for 30 Hz counter
        target_hz = 30
        time_interval = 1.0 / target_hz
        next_time = time.perf_counter()

        while not stop_event.is_set(): # Before program stop
            now = time.perf_counter()
            # Start when recording_event is triggered
            if recording_event.is_set():
                if not recording: # Initialization
                    print("+ Proprioception logger started.")
                    log = []  # Reset log
                    start_time = time.time()
                    recording = True

                # Calculate time elapsed
                timestamp = time.time()
                elapsed = timestamp - start_time

                # TCP pose
                tcp_pose_m = rtde_receive.getActualTCPPose()
                tcp_pose_mm = [x * 1000 if i < 3 else x for i, x in enumerate(tcp_pose_m)]
                # Joint positions
                joint_pos = rtde_receive.getActualQ()
                # Gripper pose
                gripper_pos = gripper.get_pos() or 0
                gripper_mm = gripper_pos_to_mm(gripper_pos)

                # Append to log
                log.append({
                    "elapsed_time": elapsed,
                    "tcp_pose_mm": tcp_pose_mm,
                    "joint_positions_rad": joint_pos,
                    "gripper_position_0_255": gripper_pos,
                    "gripper_position_mm": gripper_mm
                })

            # When recording is just stopped
            elif recording and not recording_event.is_set():
                # Save log to JSON file
                if log:
                    # Create unique filename to prevent overwriting
                    filepath = get_unique_filename("proprioception_log", ".json", root_dir)
                    # Save file
                    with open(filepath, "w") as f:
                        json.dump(log, f, indent=4)
                    print(f"! Proprioception logger stopped, data saved to {filepath}")
                else:
                    print("No proprioception data recorded.")
                recording = False
                log = [] # Reset log

            # Precise 30 Hz sleep
            next_time += time_interval
            sleep_duration = next_time - time.perf_counter()
            if sleep_duration > 0: # Sleep only if there is time remaining before next target
                time.sleep(sleep_duration)

    finally:
        print("Proprioception logger stopped.")