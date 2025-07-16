"""Sensor manager for proprioception logging, DIGIT sensor and RGB camera data acquisition for a specified duration"""
"""Uses function proprioception_logger from proprioception_logger.py"""

import os
import time
import cv2
import threading
import queue
from digit_sensor_capture import DigitHandler, Digit
from proprioception_logger import proprioception_logger
from camera_capture import camera_capture

# Set directories
root_dir = r"/home/sujatha/Desktop/" # Root directory
sub_dir = os.path.join(root_dir, "16_07_2025") # Change name accordingly
os.makedirs(sub_dir, exist_ok=True) # If no directory exists then create directory

# Set duration of capture
duration = 10 # Seconds

# Set static IP of robot
robot_ip = "192.168.1.223"

# Detect working camera index
camera_index = None

for i in range(5): # Checking for camera in indices 0-4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        cap.release()
        camera_index = i
        break

if camera_index is None:
    print("No camera detected.")

# Set up threading and queue for camera
camera_queue = queue.Queue()
camera_thread = threading.Thread(target=camera_capture, args=(sub_dir, duration, camera_index, camera_queue))

"""Starts proprioception data logging using proprioception_logger.py"""
# Set up thread for recording proprioception data in background
prop_thread = threading.Thread(target=proprioception_logger, args=(sub_dir, duration, robot_ip))
prop_thread.start()
camera_thread.start()

"""Starts DIGIT sensor data acquisition"""
# Capture DIGIT sensor data
# Subdirectory for DIGIT sensor data
digit_dir = os.path.join(sub_dir, "digit_sensors")
os.makedirs(digit_dir, exist_ok=True) # If no directory exists then create directory

# Detect connected DIGIT sensors
digits = DigitHandler.list_digits()

digit_serials = [] # Initialize list to store DIGIT sensor serial numbers
# Obtain serial numbers of detected DIGIT sensors
for digit in digits:
    serial = digit.get("serial")
    if serial and serial not in digit_serials:
        digit_serials.append(serial)

# Connect to detected DIGIT sensors
digits_connected = []
for serial in digit_serials:
    d = Digit(serial)
    d.connect()
    digits_connected.append((serial, d))

# Create subdirectory for each connected DIGIT sensor
sensor_dirs = {}
for serial in digit_serials:
    sensor_dir = os.path.join(digit_dir, serial) # Name is serial number of sensor
    os.makedirs(sensor_dir, exist_ok=True)
    sensor_dirs[serial] = sensor_dir

# Dictionary to store frame count per DIGIT sensor
frame_counts = {serial: 0 for serial in digit_serials}

# Frame capture
start_time = time.time() # Initialize timer
print(f"DIGIT data capture started. Connected Digit sensor serial numbers: {digit_serials}")

# Capture frames within duration
while time.time() - start_time < duration:
    for serial, d in digits_connected:
        frame = d.get_frame()

        # Display frames for each DIGIT sensor
        window_name = f"DIGIT Sensor {serial}"
        cv2.imshow(window_name, frame)

        # Display camera feed
        if not camera_queue.empty():
            cam_frame = camera_queue.get()
            cv2.imshow("Camera Feed", cam_frame)

        # Save frames in respective subdirectories
        frame_name = f"{serial}_{frame_counts[serial]:05d}.png" # Saves frame count in 5 digits
        frame_path = os.path.join(sensor_dirs[serial], frame_name)
        cv2.imwrite(frame_path, frame)
        frame_counts[serial] += 1

    # Update frame and quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Disconnect sensors
for _, d in digits_connected:
    d.disconnect()

# Close frame windows
cv2.destroyAllWindows()

# Print frame count
for serial, d in digits_connected:
    print(f"Total DIGIT frames ({serial}): {frame_counts[serial]:05d}")
print(f"DIGIT data capture complete. Saved to {digit_dir}.")

# Wait for proprioception thread to finish
prop_thread.join()
camera_thread.join()
print("\nAll sensors completed.")