"""To record camera data from one camera as AVI videos based on keyboard inputs"""
"""Ignores feed from RealSense cameras and DIGIT sensors"""
"""find_camera_index() works on Linux systems only"""
"""Uses function from get_unique_filename.py"""
"""Intended to be used with sensor_manager.py and associated functions"""

import cv2
import pyudev
import json
from time import perf_counter as now

from get_unique_filename import get_unique_filename


# Function to automatically detect an index where a camera is connected ignoring DIGIT sensors and RealSense cameras
def find_camera_index():
    """
    Returns first working camera index, ignoring DIGIT sensors and RealSense cameras using udev metadata
    pyudev works only on Linux systems
    """

    context = pyudev.Context() # List devices
    for device in context.list_devices(subsystem='video4linux'):
        dev_node = device.device_node
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
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read() # Open video stream
            cap.release() # Close video stream
            if ret and frame is not None:
                return index
        if not cap.isOpened():
            print(f"Failed to open camera {index}.")
    return None


# Main function to capture and preview camera data
def camera_capture(root_dir: str, recording_event, stop_event, frame_queue):
    """
        Records multiple videos from a camera in one program session based on keyboard inputs

        Args:
            root_dir: Root directory to save videos
            recording_event: threading.Event to start recording
            stop_event: threading.Event to stop the program
            frame_queue: queue.Queue to store frames for live preview
        """

    # Detect working camera automatically
    cam_index = find_camera_index()
    if cam_index is None:
        print("No camera detected.")
        return
    # print(f"Detected camera index: {cam_index}")

    # Initialize cameras
    cap = cv2.VideoCapture(cam_index) # Open video stream
    if not cap.isOpened():
        print(f"Failed to open camera index {cam_index}")
        return

    # Initialize variables
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    start_time = None
    frame_log = [] # To store frame timestamps
    frame_count = 0
    frame_buffer = [] # To store frames in memory till recording stops
    recording_initialized = False # Flag

    # Main loop
    try:
        print(f"- Camera {cam_index} thread ready.")
        while not stop_event.is_set(): # Before program stop
            ret, frame = cap.read() # Get frame
            if not ret:
                continue # Continue even if frame is skipped

            # Live preview of camera feeds even when not recording
            if frame_queue and not frame_queue.full():
                frame_queue.put(frame)

            # Start recording when triggered
            if recording_event.is_set():
                # Initialization
                if not recording_initialized:
                    start_time = now() # Start timer
                    # Reset variables
                    frame_log = []
                    frame_count = 0
                    frame_buffer = []
                    recording_initialized = True # Set flag
                    print(f"+ Camera {cam_index} recording started.")

                # Save frames within recording duration
                frame_buffer.append(frame)
                # Log frame time
                time_elapsed = now() - start_time
                frame_log.append({
                    "elapsed_time": time_elapsed,
                    "frame_count": frame_count
                })
                frame_count += 1

            # Stop recording when recording_event is cleared
            elif recording_initialized:
                print(f"! Camera {cam_index} recording stopped.")

                # Save frame log
                # Get unique filepath to prevent overwriting
                log_path = get_unique_filename(f"camera_{cam_index}_log", ".json", root_dir)
                with open(log_path, "w") as f:
                    json.dump(frame_log, f, indent=4)

                # Calculate actual FPS
                duration = now() - start_time
                actual_fps = frame_count / duration if duration > 0 else 30

                # Catch FPS deviations greater than 3
                if abs(actual_fps - 30) > 3:
                    print(f"[WARNING] Camera {cam_index} FPS deviated significantly: {actual_fps:.2f}")

                # Write frames using actual FPS
                # Get unique filepath to prevent overwriting
                filepath = get_unique_filename(f"camera_{cam_index}", ".avi", root_dir)
                # Get frame properties
                height, width = frame_buffer[0].shape[:2]

                # Write frames
                frame_writer = cv2.VideoWriter(filepath, fourcc, actual_fps, (width, height))
                for frame in frame_buffer:
                    frame_writer.write(frame)
                frame_writer.release()

                print(f"Camera {cam_index} video saved with {actual_fps:.2f} FPS to {filepath}")

                # Reset variables
                start_time = None
                frame_log = []
                frame_count = 0
                frame_buffer = []
                recording_initialized = False

    finally:
        # Close video streams
        cap.release()
        print("Camera capture stopped.")