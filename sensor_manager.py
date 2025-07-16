"""Sensor manager for proprioception, DIGIT sensor and RGB camera data acquisition for a specified duration"""
"""Uses functions proprioception_logger from proprioception_logger.py and camera_capture from camera_capture.py"""

import os
import time
import cv2
import threading
import queue
from digit_sensor_capture import DigitHandler, Digit
from proprioception_logger import proprioception_logger
from camera_capture import camera_capture

# Set parameters
root_dir = r"/home/sujatha/PycharmProjects/UR3e-Python-Teleop/Trial" # Root directory
os.makedirs(root_dir, exist_ok=True) # Create directory if it does not exist

duration = 30 # Duration of capture in seconds
robot_ip = "192.168.1.223" # Static IP address of robot


# Function to obtain working camera index

def find_camera_index(max_index=5):
    """
    Loops through indices to find a working camera index
    Assumes only one camera is connected
    """
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None


# Obtain working camera index
camera_index = find_camera_index()
if camera_index is None: # Print if no camera detected
    print("No camera detected.")

# Set up threading and queue for camera
camera_queue = queue.Queue()
camera_thread = threading.Thread(target=camera_capture, args=(root_dir, duration, camera_index, camera_queue))

# Set up thread for recording proprioception data
prop_thread = threading.Thread(target=proprioception_logger, args=(root_dir, duration, robot_ip))

# Start threads
prop_thread.start()
camera_thread.start()


# Function to obtain DIGIT sensor serial numbers
def find_digit_serials():
    """
    Returns a list of serial numbers of detected DIGIT sensors.
    """
    digits = DigitHandler.list_digits()
    serials = []
    for digit in digits:
        serial = digit.get("serial")
        if serial and serial not in serials:
            serials.append(serial)
    return serials


# Obtain DIGIT sensor serial numbers
digit_serials = find_digit_serials()
if not digit_serials: # Print if no DIGIT sensor detected
    print("No DIGIT sensor detected.")

# Connect to detected DIGIT sensors
digits_connected = []
for serial in digit_serials:
    d = Digit(serial)
    d.connect()
    digits_connected.append((serial, d)) # Obtain tuple of connected DIGIT sensors

# Define VideoWriters for DIGIT sensors
digit_writers = {}
fps = 30 # Adjust frame rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Frame capture
frame_count = {}
first_frame_size = {}
start_time = time.time() # Initialize timer
print(f"DIGIT data capture started. Connected Digit sensor serial numbers: {digit_serials}")

# Capture frames within duration
while time.time() - start_time < duration:
    for serial, d in digits_connected:
        frame = d.get_frame() # Get frame

        # Get frame size from first frame of each sensor
        if serial not in first_frame_size:
            height, width = frame.shape[:2]
            first_frame_size[serial] = (width, height)

            # Set filename for DIGIT sensor videos
            video_path = os.path.join(root_dir, f"{serial}_output.avi")
            digit_writers[serial] = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Initialize frame count
        if serial not in frame_count:
            frame_count[serial] = 0

        # Display live preview for each DIGIT sensor
        cv2.imshow(f"DIGIT Sensor {serial}", frame)

        # Display live preview for camera
        if not camera_queue.empty():
            cam_frame = camera_queue.get()
            cv2.imshow("Camera Feed", cam_frame)

        # Write frames to corresponding DIGIT sensor videos
        digit_writers[serial].write(frame)
        frame_count[serial] += 1

    # Update frame and quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Disconnect sensors
for _, d in digits_connected:
    d.disconnect()

# Release DIGIT VideoWriters
for writer in digit_writers.values():
    writer.release()

# Close frame windows
cv2.destroyAllWindows()

# Print frame count
for serial in digit_serials:
    print(f"DIGIT video saved ({frame_count[serial]} frames): {serial}_output.avi")
print(f"DIGIT data capture complete. Saved to {root_dir}")

# Wait for proprioception thread to finish
prop_thread.join()
camera_thread.join()
print("\nAll sensors completed.")