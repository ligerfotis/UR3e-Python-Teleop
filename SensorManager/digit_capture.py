"""Function to record DIGIT sensor data from multiple connected sensors as AVI videos based on keyboard inputs"""
"""Uses function from get_unique_filename.py"""
"""Intended to be used with sensor_manager.py and associated functions"""

import threading
import cv2
import time
import json
from digit_interface import Digit
from digit_interface.digit_handler import DigitHandler

from get_unique_filename import get_unique_filename


def digit_capture(root_dir: str, recording_event, stop_event, frame_queues: dict):
    """
    Start threads for multiple DIGIT sensors in one program session based on keyboard inputs

    Args:
        root_dir: Root directory to save videos
        recording_event: threading.Event to start recording
        stop_event: threading.Event to stop the program
        frame_queues: queue.Queue per DIGIT sensor to store frames for live preview
    """

    # Detect connected DIGIT sensors
    raw_digits = DigitHandler.list_digits()
    unique_digits = {d['serial']: d for d in raw_digits}.keys() # Only add unique serial numbers

    if not unique_digits:
        print("No DIGIT sensors detected.")
        return

    # Create separate threads for each DIGIT sensor
    threads = []
    for serial in unique_digits:
        queue = frame_queues.get(serial)
        t = threading.Thread(
            target=run_digit_thread,
            args=(serial, root_dir, recording_event, stop_event, queue)
        )
        t.start() # Start threads
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()
    print("DIGIT capture stopped.")


def run_digit_thread(serial, root_dir, recording_event, stop_event, frame_queue):
    """
    Runs thread for one DIGIT sensor to record and live preview video data

    Args:
        serial: Serial number of DIGIT sensor
        root_dir: Root directory to save videos
        recording_event: threading.Event to start recording
        stop_event: threading.Event to stop the program
        frame_queue: queue.Queue for DIGIT sensor to store frames for live preview
    """

    # Connect to DIGIT sensor
    try:
        digit = Digit(serial)
        digit.connect()
        # print(f"Connected to DIGIT {serial}")
    except Exception as e:
        print(f"Failed to connect to DIGIT {serial}: {e}")
        return

    # Initialize variables
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # VideoWriter for AVI
    fps = 30 # Writer FPS
    digit_writer = None
    frame_log = []
    frame_count = 0
    start_time = None

    # Main loop
    try:
        print(f"- DIGIT {serial} thread ready.")
        while not stop_event.is_set(): # Before program stop
            try:
                frame = digit.get_frame()
            except Exception as e:
                print(f"Cannot retrieve frame from DIGIT {serial}: {e}")
                continue # Continue even if frame is skipped

            # Live preview of camera feeds even when not recording
            if frame_queue and not frame_queue.full():
                frame_queue.put(frame)

            # Start recording when triggered
            if recording_event.is_set():
                if digit_writer is None: # Initialization
                    # Get camera properties
                    h, w = frame.shape[:2]
                    # Get unique filepath to prevent overwriting
                    filepath = get_unique_filename(f"{serial}_output", ".avi", root_dir)
                    # Define VideoWriter
                    digit_writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
                    # Initialize for frame logging
                    frame_log = []
                    start_time = time.time()
                    frame_count = 0
                    print(f"+ DIGIT {serial} recording started, saving to {filepath}")

                # Write frames within recording duration
                digit_writer.write(frame)
                # Log frame time
                elapsed = time.time() - start_time
                frame_log.append({
                    "elapsed_time": elapsed,
                    "frame_count": frame_count
                })
                frame_count += 1

            # Stop recording when recording_event is cleared
            elif digit_writer:
                digit_writer.release()
                digit_writer = None
                print(f"! DIGIT {serial} recording stopped.")

                # Save log
                log_path = get_unique_filename(f"{serial}_log", ".json", root_dir)
                with open(log_path, "w") as f:
                    json.dump(frame_log, f, indent=4)
                frame_log = [] # Reset log

    # Release connected DIGIT sensors and VideoWriters
    finally:
        if digit_writer:
            digit_writer.release()
        digit.disconnect()
        print(f"DIGIT {serial} thread exited.")