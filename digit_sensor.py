import os
import time
from digit_interface import DigitHandler, Digit
import cv2

# Configure root_directory
root_dir = "sensor_data"

# Subdirectory for DIGIT sensor data
digit_dir = os.path.join(root_dir, "digit_sensor")
os.makedirs(digit_dir, exist_ok=True) # If no directory exists then create directory

# Duration of capture
duration = 20 # Seconds

# Detect connected DIGIT sensors
digits = DigitHandler.list_digits()

digit_serials = [] # Initialize list to store DIGIT sensor serial numbers
# Obtain serial numbers of detected DIGIT sensors
for digit in digits:
    serial = digit.get("serial")
    if serial and serial not in digit_serials:
        digit_serials.append(serial)

# Print serial numbers of detected DIGIT sensors
print("Connected Digit sensor serial numbers:", digit_serials)

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
print(f"Capturing frames for {duration} seconds.")

# Capture frames within duration
while time.time() - start_time < duration:
    for serial, d in digits_connected:
        frame = d.get_frame()

        # Display frames for each DIGIT sensor
        window_name = f"DIGIT Sensor {serial}"
        cv2.imshow(window_name, frame)

        # Save frames in respective subdirectories
        frame_name = f"{serial}_{frame_counts[serial]:04d}.png" # Saves frame count in 4 digits
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
    print(f"Total frames ({serial}): {frame_counts[serial]:04d}")
print("Frame capture complete.")
