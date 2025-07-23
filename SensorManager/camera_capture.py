"""Function to record camera data from multiple cameras as AVI videos based on keyboard inputs"""
"""Ignores feed from RealSense cameras and DIGIT sensors"""
"""find_camera_indices() works on Linux systems only"""
"""Uses function from get_unique_filename.py"""
"""Intended to be used with sensor_manager.py and associated functions"""

import cv2
import pyudev
import time
import json

from get_unique_filename import get_unique_filename


# Function to automatically detect indices where cameras are connected ignoring DIGIT sensors and RealSense cameras
def find_camera_indices():
    """
    Returns list of real camera indices, ignoring DIGIT sensors and RealSense cameras using udev metadata
    pyudev works only on Linux systems
    """
    context = pyudev.Context() # List devices
    camera_indices = []

    # Identify connected video output devices
    for device in context.list_devices(subsystem='video4linux'):
        dev_node = device.device_node # Gets filepath of video output devices
        if not dev_node.startswith("/dev/video"):
            continue

        # Extract camera index
        index = int(dev_node.replace("/dev/video", ""))
        # Check parent USB device info
        parent = device.find_parent('usb', 'usb_device')
        if parent:
            model = parent.get('ID_MODEL', '').lower()
            vendor = parent.get('ID_VENDOR', '').lower()
            # Filter out DIGIT sensors and RealSense cameras
            if 'digit' in model or 'realsense' in model or 'intel' in vendor or 'd435' in model:
                continue

        # Confirm camera is working
        print(f"Checking /dev/video{index} for non-RealSense camera...")
        cap = cv2.VideoCapture(index) # Open video stream
        if cap.isOpened():
            ret, frame = cap.read() # Read frame
            if ret and frame is not None:
                camera_indices.append(index)
            cap.release() # Close video stream

    return sorted(camera_indices)


# Main function to capture and preview camera data
def camera_capture(root_dir: str, recording_event, stop_event, frame_queues: dict):
    """
    Records multiple videos in one program session based on keyboard inputs

    Args:
        root_dir: Root directory to save videos
        recording_event: threading.Event to start recording
        stop_event: threading.Event to stop the program
        frame_queues: queue.Queue per camera index to store frames for live preview
    """

    # Detect working cameras ignoring DIGIT sensors
    camera_indices = find_camera_indices()
    if not camera_indices:
        print("No camera indices detected.")
        return
    else:
        print(f"Detected camera indices: {camera_indices}")

    caps = []
    # Initialize cameras
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx) # Open video streams
        if cap.isOpened():
            caps.append((idx, cap))
        else:
            print(f"Failed to open camera index {idx}")

    # Initialize variables
    cam_writers = {}
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # VideoWriter for AVI
    fps = 30 # Writer FPS
    start_times = {}
    frame_logs = {}
    frame_counts = {}

    # Main loop
    try:
        print("- Camera thread ready.")
        while not stop_event.is_set(): # Before program stop
            for idx, cap in caps:
                ret, frame = cap.read() # Read frames
                if not ret: # Continue even if frame is missed
                    continue

                # Live preview of camera feeds even when not recording
                if idx in frame_queues and not frame_queues[idx].full():
                    frame_queues[idx].put(frame)

                # Start recording when triggered
                if recording_event.is_set():
                    if idx not in cam_writers: # Initialization
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        # Get unique filepath to prevent overwriting
                        filepath = get_unique_filename(f"camera_{idx}", ".avi", root_dir)
                        # Define VideoWriter
                        cam_writers[idx] = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                        # Initialize for frame logging
                        start_times[idx] = time.time()
                        frame_logs[idx] = []
                        frame_counts[idx] = 0
                        print(f"+ Camera {idx} recording started. Saving to {filepath}")

                    # Write frames within recording duration
                    cam_writers[idx].write(frame)
                    # Log frame time
                    elapsed_time = time.time() - start_times[idx]
                    frame_logs[idx].append({
                        "elapsed_time": elapsed_time,
                        "frame_count": frame_counts[idx],
                    })
                    frame_counts[idx] += 1

                # Stop recording when recording_event is cleared
                elif idx in cam_writers:
                    cam_writers[idx].release()
                    del cam_writers[idx]
                    print(f"! Camera {idx} recording stopped.")

                    # Save log
                    log_path = get_unique_filename(f"camera_{idx}_log", ".json", root_dir)
                    with open(log_path, "w") as f:
                        json.dump(frame_logs[idx], f, indent=4)
                    print(f"! Camera {idx} frame log saved to {log_path}")
                    frame_logs[idx] = [] # Reset log

    # Release video streams and VideoWriters
    finally:
        for idx, cap in caps:
            cap.release()
        for writer in cam_writers.values():
            writer.release()
        print("Camera capture stopped.")