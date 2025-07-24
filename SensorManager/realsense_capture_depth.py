"""Function to capture RealSense depth data as AVI videos based on keyboard inputs"""
"""Tested with RealSense D435"""
"""Uses function from get_unique_filename.py"""
"""Intended to be used with sensor_manager.py and associated functions"""

import pyrealsense2 as rs
import cv2
import numpy as np
import time
import json
from get_unique_filename import get_unique_filename


def realsense_capture_depth(root_dir: str, recording_event, stop_event, frame_queue=None):
    """
            Records multiple RealSense depth videos in one program session based on keyboard inputs

            Args:
                root_dir: Root directory to save videos
                recording_event: threading.Event to start recording
                stop_event: threading.Event to stop the program
                frame_queue: queue.Queue to store frames for live preview
            """

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable stream
    # Default width and height (640, 480), 16-bit depth channels, 30 FPS
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start pipeline
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Failed to start RealSense depth pipeline: {e}")
        return

    # Initialize variables
    frame_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # VideoWriter for AVI
    fps = 30 # Writer FPS
    start_time = None
    frame_log = []
    frame_count = 0

    # Main loop
    try:
        print("- RealSense depth thread ready.")
        while not stop_event.is_set(): # Before program stop
            # Wait for a coherent frame pair
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame: # Continue even if frame is missed
                continue

            # Convert to numpy arrays for display
            depth_image = np.asanyarray(depth_frame.get_data())
            # Apply colormap
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            if frame_queue and not frame_queue.full():
                frame_queue.put(depth_colormap)

            # Start recording when triggered
            if recording_event.is_set():
                if frame_writer is None: # Initialization
                    # Get camera properties
                    height, width = depth_colormap.shape[:2]
                    # Get unique filepath to prevent overwriting
                    filepath = get_unique_filename("realsense_depth", ".avi", root_dir)
                    # Define VideoWriter
                    frame_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                    # Initialize for frame logging
                    start_time = time.time()
                    frame_log = []
                    frame_count = 0
                    print(f"+ RealSense depth recording started, saving to {filepath}")

                # Write frames within recording duration
                frame_writer.write(depth_colormap)
                # Log frame time
                elapsed_time = time.time() - start_time
                frame_log.append({
                    "elapsed_time": elapsed_time,
                    "frame_count": frame_count
                })
                frame_count += 1

            # Stop recording when recording_event is cleared
            elif frame_writer:
                frame_writer.release()
                frame_writer = None
                print("! RealSense depth recording stopped.")

                # Save log
                if frame_log:
                    log_path = get_unique_filename("realsense_depth_log", ".json", root_dir)
                    with open(log_path, "w") as f:
                        json.dump(frame_log, f, indent=4)
                    frame_log = [] # Reset log

    # Release video streams and VideoWriters
    finally:
        pipeline.stop()
        if frame_writer:
            frame_writer.release()
        print("RealSense depth thread stopped.")