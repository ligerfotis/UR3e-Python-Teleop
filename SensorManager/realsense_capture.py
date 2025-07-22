"""Function to capture RealSense RGB and depth data as AVI videos based on keyboard inputs"""
"""Tested with RealSense D435"""
"""Uses function from get_unique_filename.py"""
"""Intended to be used with sensor_manager.py and associated functions"""

import pyrealsense2 as rs
import cv2
import numpy as np
import time

from get_unique_filename import get_unique_filename


def realsense_capture(root_dir: str, recording_event, stop_event, frame_queues: dict):
    """
    Records multiple RealSense RGB and depth videos in one program session based on keyboard inputs

    Args:
        root_dir: Root directory to save videos
        recording_event: threading.Event to start recording
        stop_event: threading.Event to stop the program
        frame_queues: queue.Queue per camera index to store frames for live preview
    """

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams
    # Default width and height (640, 480), 8-bit channels, 30 FPS
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # RGB
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # Depth

    # Start pipeline
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Failed to start RealSense pipeline: {e}")
        return

    color_writer, depth_writer = None, None
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # VideoWriter for AVI
    fps = 30 # Writer FPS

    # Main loop
    try:
        print("- RealSense thread ready.")
        while not stop_event.is_set(): # Before program stop
            # Wait for a coherent frame pair
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame: # Continue even if frame is missed
                continue

            # Convert to numpy arrays for display
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Live preview of camera feeds even when not recording
            if "color" in frame_queues and not frame_queues["color"].full():
                frame_queues["color"].put(color_image)
            if "depth" in frame_queues and not frame_queues["depth"].full():
                frame_queues["depth"].put(depth_colormap)

            # Start recording when triggered
            if recording_event.is_set():
                if color_writer is None or depth_writer is None: # Initialization
                    # Get camera properties
                    height, width = color_image.shape[:2]
                    # Get unique filepath to prevent overwriting
                    color_path = get_unique_filename("realsense_color", ".avi", root_dir)
                    depth_path = get_unique_filename("realsense_depth", ".avi", root_dir)
                    # Define VideoWriters
                    color_writer = cv2.VideoWriter(color_path, fourcc, fps, (width, height))
                    depth_writer = cv2.VideoWriter(depth_path, fourcc, fps, (width, height))
                    print(f"+ RealSense recording started: \n  Color video saving to {color_path}\n  Depth video saving to {depth_path}")

                # Write frames within recording duration
                color_writer.write(color_image)
                depth_writer.write(depth_colormap)

            # Stop recording when recording_event is cleared
            elif color_writer or depth_writer:
                if color_writer:
                    color_writer.release()
                    color_writer = None
                    print("! RealSense color recording stopped.")
                if depth_writer:
                    depth_writer.release()
                    depth_writer = None
                    print("! RealSense depth recording stopped.")

            time.sleep(0.03) # Idle time to wait for next recording

    # Release video streams and VideoWriters
    finally:
        pipeline.stop()
        if color_writer:
            color_writer.release()
        if depth_writer:
            depth_writer.release()
        print("RealSense capture stopped.")
