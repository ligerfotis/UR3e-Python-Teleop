"""Function to record DIGIT sensor data from multiple connected sensors as AVI videos based on keyboard inputs"""
"""Uses function from get_unique_filename.py"""
"""Intended to be used with sensor_manager.py and associated functions"""

import cv2
from digit_interface import Digit
from digit_interface.digit_handler import DigitHandler

from SensorManager.get_unique_filename import get_unique_filename


def digit_capture(root_dir: str, recording_event, stop_event, frame_queues: dict):
    """
    Records multiple videos in one program session based on keyboard inputs

    Args:
        root_dir: Root directory to save videos
        recording_event: threading.Event to start recording
        stop_event: threading.Event to stop the program
        frame_queues: queue.Queue per camera index to store frames for live preview
    """

    # Detect connected DIGIT sensors
    raw_digits = DigitHandler.list_digits()
    seen_serials = set()
    unique_digits = []
    for d in raw_digits:
        if d["serial"] not in seen_serials: # Only add unique serial numbers
            unique_digits.append(d)
            seen_serials.add(d["serial"])

    if not unique_digits:
        print("No DIGIT sensors detected.")
        return

    digits = {}
    # Connect to each DIGIT sensor
    for d_info in unique_digits:
        serial = d_info["serial"]
        try:
            d = Digit(serial)
            d.connect()
            digits[serial] = d
            print(f"Connected to DIGIT sensor {serial}")
        except Exception as e:
            print(f"Failed to connect to DIGIT {serial}: {e}")

    # Initialize variables
    digit_writers = {}
    fourcc = cv2.VideoWriter_fourcc(*"XVID") # VideoWriter for AVI
    fps = 30 # Writer FPS

    # Main loop
    try:
        print("- DIGIT thread ready.")
        while not stop_event.is_set(): # Before program stop
            for serial, d in digits.items():
                try:
                    frame = d.get_frame()
                except Exception as e:
                    print(f"Cannot retrieve frame from DIGIT {serial}, skipping: {e}")
                    continue

                # Live preview of camera feeds even when not recording
                if serial in frame_queues and not frame_queues[serial].full():
                    frame_queues[serial].put(frame)

                # Start recording when triggered
                if recording_event.is_set():
                    if serial not in digit_writers:
                        # Get camera properties
                        height, width = frame.shape[:2]
                        # Get unique filepath to prevent overwriting
                        filepath = get_unique_filename(f"{serial}_output", ".avi", root_dir)
                        # Define VideoWriter
                        digit_writers[serial] = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                        print(f"+ DIGIT {serial} recording started. Saving to {filepath}")

                    # Write frames within recording duration
                    digit_writers[serial].write(frame)

                # Stop recording when recording_event is cleared
                elif serial in digit_writers:
                    digit_writers[serial].release()
                    del digit_writers[serial]
                    print(f"! DIGIT {serial} recording stopped.")

    # Release connected DIGIT sensors and VideoWriters
    finally:
        for d in digits.values():
            d.disconnect()
        for writer in digit_writers.values():
            writer.release()
        print("DIGIT capture stopped.")