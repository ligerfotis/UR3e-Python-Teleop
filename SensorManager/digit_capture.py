"""To record DIGIT sensor data from multiple connected sensors as AVI videos based on keyboard inputs"""
"""Uses function from get_unique_filename.py"""
"""Intended to be used with sensor_manager.py and associated functions"""

import threading
import cv2
import json
from digit_interface import Digit
from digit_interface.digit_handler import DigitHandler
from time import perf_counter as now

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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    start_time = None
    frame_log = [] # To store frame timestamps
    frame_count = 0
    frame_buffer = [] # To store frames in memory till recording stops
    recording_initialized = False # Flag

    # Main loop
    try:
        print(f"- DIGIT {serial} thread ready.")
        while not stop_event.is_set(): # Before program stop
            try:
                frame = digit.get_frame() # Get frame
            except Exception as e:
                print(f"Cannot retrieve frame from DIGIT {serial}: {e}")
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
                    print(f"+ DIGIT {serial} recording started.")

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
                print(f"! DIGIT {serial} recording stopped.")

                # Save log
                # Get unique filepath to prevent overwriting
                log_path = get_unique_filename(f"{serial}_log", ".json", root_dir)
                with open(log_path, "w") as f:
                    json.dump(frame_log, f, indent=4)

                # Calculate actual FPS
                duration = now() - start_time
                actual_fps = frame_count / duration if duration > 0 else 30

                # Catch FPS deviations greater than 3
                if abs(actual_fps - 30) > 3:
                    print(f"[WARNING] DIGIT {serial} FPS deviated significantly: {actual_fps:.2f}")

                # Write frames using actual FPS
                # Get unique filepath to prevent overwriting
                filepath = get_unique_filename(f"digit_{serial}", ".avi", root_dir)
                # Get frame properties
                height, width = frame_buffer[0].shape[:2]

                # Write frames
                frame_writer = cv2.VideoWriter(filepath, fourcc, actual_fps, (width, height))
                for frame in frame_buffer:
                    frame_writer.write(frame)
                frame_writer.release()

                print(f"DIGIT {serial} video saved with {actual_fps:.2f} FPS to {filepath}")

                # Reset variables
                start_time = None
                frame_log = []
                frame_count = 0
                frame_buffer = []
                recording_initialized = False

    finally:
        # Disconnect DIGIT sensors
        digit.disconnect()
        print(f"DIGIT {serial} thread exited.")