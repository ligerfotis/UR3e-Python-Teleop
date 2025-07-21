"""Function to record DIGIT sensor data from multiple connected sensors as AVI videos"""
"""Uses function from get_unique_filename.py"""
"""Intended to be used with sensor_manager.py for coordinated live preview of sensors with other sensors"""

import cv2
from digit_interface import Digit
from digit_interface.digit_handler import DigitHandler

from SensorManager.get_unique_filename import get_unique_filename


def digit_capture(root_dir: str, start_event, stop_event, frame_queues: dict):
    """
    Args:
        root_dir: Root directory to save videos
        start_event: threading.Event to start recording
        stop_event: threading.Event to stop recording
        frame_queues: queue.Queue per DIGIT serial to store frames for live preview
    """

    # Detect connected DIGIT sensors
    raw_digits = DigitHandler.list_digits()
    seen_serials = set()
    detected_digits = []

    for d in raw_digits:
        serial = d["serial"]
        if serial not in seen_serials:
            detected_digits.append(d)
            seen_serials.add(serial)

    if not detected_digits:
        print("No unique DIGIT sensors detected.")
        return

    # Initialize variables
    digits_connected = {}
    digit_writers = {}
    fourcc = cv2.VideoWriter_fourcc(*"XVID") # VideoWriter for AVI
    fps = 30 # Writer FPS

    # Connect to each DIGIT sensor
    for d_info in detected_digits:
        serial = d_info["serial"]
        try:
            d = Digit(serial)
            d.connect()
            digits_connected[serial] = d
            print(f"Connected to DIGIT sensor {serial}")
        except Exception as e:
            print(f"Failed to connect to DIGIT {serial}: {e}")

    # Main loop
    while not stop_event.is_set(): # Before recording stop
        for serial, d in digits_connected.items():
            frame = d.get_frame() # Read frames

            # Live preview of DIGIT feeds even when not recording
            if serial in frame_queues:
                if not frame_queues[serial].full(): # Add frames to queues if not full
                    frame_queues[serial].put((serial, frame))

            # Start recording
            if start_event.is_set() and serial not in digit_writers: # Initialization
                # Get camera properties
                height, width = frame.shape[:2]
                # Get unique filepath to prevent overwriting
                filepath = get_unique_filename(f"{serial}_output", ".avi", root_dir)
                # Define VideoWriter
                digit_writers[serial] = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                print(f"Recording started for DIGIT {serial} at {filepath}")

            # Write frames within recording duration
            if start_event.is_set() and serial in digit_writers:
                digit_writers[serial].write(frame)

    # Disconnect DIGIT sensors and release VideoWriters
    for d in digits_connected.values():
        d.disconnect()
    for writer in digit_writers.values():
        writer.release()
    print("DIGIT capture stopped.")