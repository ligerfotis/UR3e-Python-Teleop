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
from realsense_capture_color import realsense_capture_color
from realsense_capture_depth import realsense_capture_depth
from audio_capture import audio_capture
from proprioception_logger import proprioception_logger

# Set parameters
robot_ip = "192.168.1.223" # Ensure static IP of robot matches that of computer
root_dir = r"/home/sujatha/Test3" # Root directory to save all recorded data
os.makedirs(root_dir, exist_ok=True)

# Set recording control events
recording_event = threading.Event()
stop_event = threading.Event()

print("Waiting for sensors to start...")

# Thread for keyboard control of recording
keyboard_thread = threading.Thread(
    target=sensor_control_keyboard,
    args=(recording_event, stop_event),
    daemon=True
)
keyboard_thread.start() # Start keyboard thread

# Queues and thread for camera capture
camera_frame_queue = queue.Queue(maxsize=1)

camera_thread = threading.Thread(
    target=camera_capture,
    args=(root_dir, recording_event, stop_event, camera_frame_queue),
)
camera_thread.start() # Start camera thread

# Queues and thread for DIGIT capture
digit_devices = DigitHandler.list_digits()
digit_frame_queues = {}  # Dict => DIGIT Serial: queue.Queue
for d_info in digit_devices:
    serial = d_info["serial"]
    digit_frame_queues[serial] = queue.Queue(maxsize=1) # Queues for all detected DIGIT sensors

digit_thread = threading.Thread(
    target=digit_capture,
    args=(root_dir, recording_event, stop_event, digit_frame_queues),
)
digit_thread.start() # Start DIGIT thread

# Queues and thread for RealSense capture
realsense_queues = {
    "color": queue.Queue(maxsize=5),
    "depth": queue.Queue(maxsize=5)
}

realsense_color_thread = threading.Thread(
    target=realsense_capture_color,
    args=(root_dir, recording_event, stop_event, realsense_queues["color"])
)
realsense_depth_thread = threading.Thread(
    target=realsense_capture_depth,
    args=(root_dir, recording_event, stop_event, realsense_queues["depth"])
)
realsense_color_thread.start() # Start RealSense threads
realsense_depth_thread.start()

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
while not stop_event.is_set():
    # Show camera frames
    if not camera_frame_queue.empty():
        frame = camera_frame_queue.get()
        cv2.imshow("Camera", frame)

    # Show DIGIT frames
    for serial, digit_queue in digit_frame_queues.items():
        if not digit_queue.empty():
            frame = digit_queue.get()
            if frame is not None:
                cv2.imshow(f"DIGIT {serial}", frame)

    # Show RealSense frames
    for stream, q in realsense_queues.items():
        if not q.empty():
            frame = q.get()
            if frame is not None:
                cv2.imshow(f"RealSense {stream}", frame)

    # Exit if 's' is pressed
    if cv2.waitKey(10) & 0xFF == ord("3"):
        stop_event.set()
        break

# Wait for all threads to stop
camera_thread.join()
digit_thread.join()
realsense_color_thread.join()
realsense_depth_thread.join()
audio_thread.join()
prop_thread.join()
keyboard_thread.join()

# Close live preview
cv2.destroyAllWindows()
print("All sensors stopped.")