"""Sensor manager for capturing data from multiple sensors and logging proprioception of a UR cobot"""
"""Includes functions from sensor_control_keyboard.py, proprioception_logger.py , camera_capture.py, digit_capture.py, get_unique_filename.py"""
"""Tested using a UR3e cobot"""
"""Run as sudo due to keyboard control of recording"""

import os
import threading
import queue
import cv2
from digit_interface import DigitHandler

from sensor_control_keyboard import sensor_control_keyboard
from camera_capture import camera_capture
from digit_capture import digit_capture
from realsense_capture import realsense_capture
from audio_capture import audio_capture
from proprioception_logger import proprioception_logger

# Set parameters
robot_ip = "192.168.1.223" # Ensure static IP of robot matches that of computer
root_dir = r"/home/sujatha/Demo_Battery_Insertion_4"  # Root directory to save all recorded data
os.makedirs(root_dir, exist_ok=True)

# Set recording control events
recording_event = threading.Event()
stop_event = threading.Event()

# Thread for keyboard control of recording
keyboard_thread = threading.Thread(
    target=sensor_control_keyboard,
    args=(recording_event, stop_event),
    daemon=True
)
keyboard_thread.start() # Start keyboard thread

# Queues and thread for camera capture
camera_frame_queues = {}  # Dict => Camera index: queue.Queue
for idx in range(15): # Queues for upto 5 cameras
    camera_frame_queues[idx] = queue.Queue(maxsize=10) # Queue holds 10 frames at a time

camera_thread = threading.Thread(
    target=camera_capture,
    args=(root_dir, recording_event, stop_event, camera_frame_queues),
)
camera_thread.start() # Start camera thread

# Queues and thread for DIGIT capture
digit_devices = DigitHandler.list_digits()
digit_frame_queues = {}  # Dict => DIGIT Serial: queue.Queue
for d_info in digit_devices:
    serial = d_info["serial"]
    digit_frame_queues[serial] = queue.Queue(maxsize=10) # Queues for all detected DIGIT sensors

digit_thread = threading.Thread(
    target=digit_capture,
    args=(root_dir, recording_event, stop_event, digit_frame_queues),
)
digit_thread.start() # Start DIGIT thread

# Queues and thread for RealSense capture
realsense_queues = {
    "color": queue.Queue(maxsize=1),
    "depth": queue.Queue(maxsize=1),
} # Dict => RealSense Camera Type: queue.Queue

realsense_thread = threading.Thread(
    target=realsense_capture,
    args=(root_dir, recording_event, stop_event, realsense_queues),
    daemon=True
)
realsense_thread.start() # Start RealSense thread

# Thread for audio capture
audio_thread = threading.Thread(
    target=audio_capture,
    args=(root_dir, recording_event, stop_event),
)
audio_thread.start()

# Thread for proprioception logging
prop_thread = threading.Thread(
    target=proprioception_logger,
    args=(root_dir, recording_event, stop_event, robot_ip),
)
prop_thread.start() # Start proprioception logging thread

# Live preview loop
print("Waiting for sensors to start...")
while not stop_event.is_set():
    # Show camera frames
    for cam_id, cam_queue in camera_frame_queues.items():
        if not cam_queue.empty(): # Gets frame if one is available in queue
            frame = cam_queue.get()
            cv2.imshow(f"Camera {cam_id}", frame)

    # Show DIGIT frames
    for serial, digit_queue in digit_frame_queues.items():
        if not digit_queue.empty():
            frame = digit_queue.get()
            if frame is not None:
                cv2.imshow(f"DIGIT {serial}", frame)

    # Show RealSense frames
    for type, realsense_queue in realsense_queues.items():
        if not realsense_queue.empty():
            frame = realsense_queue.get()
            if frame is not None:
                cv2.imshow(f"RealSense {type}", frame)

    # Exit if 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord("3"):
        stop_event.set()
        break

# Wait for all threads to stop
camera_thread.join()
digit_thread.join()
realsense_thread.join()
audio_thread.join()
prop_thread.join()

# Close live preview
cv2.destroyAllWindows()
print("All sensors stopped.")