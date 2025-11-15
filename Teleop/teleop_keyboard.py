"""Python code to control UR 6-DOF TCP movement and gripper opening/closing with a keyboard"""
"""Tested using a UR3e cobot"""
"""Run code as sudo"""
"""Gripper optional: script falls back to robot-only control if absent"""

import keyboard
from rtde_control import RTDEControlInterface
from robotiq_socket_gripper import RobotiqSocketGripper

# Connect to robot and gripper
robot_ip = "192.168.1.201" # Ensure static IP addresses of robot and computer match
rtde_control = RTDEControlInterface(robot_ip)

gripper = None
gripper_enabled = False
try:
    gripper = RobotiqSocketGripper(robot_ip)
    gripper.connect()
    gripper_enabled = True
except OSError as exc:
    print(f"Warning: Gripper unavailable ({exc}). Continuing without gripper control.")
    gripper = None

"""
UNITS:
Linear velocity -> m/s
Angular velocity -> rad/s

Speed vector -> 
[linear_velocity_x, linear_velocity_y, linear_velocity_z, angular_velocity_x, angular_velocity_y, angular_velocity_z]]
"""

# Speed parameters
linear_speed_magnitude = 0.05 # m/s
angular_speed_magnitude = 0.25 # rad/s
speed_vector = [0.0] * 6 # Initialization

# Gripper parameters
i = 1 # Increment for gripper movement
if gripper_enabled:
    gripper.set_force(20) # Range: 0 (min) to 255 (max)
    gripper.set_speed(50) # Range: 0 (min) to 255 (max)
    current_pos = gripper.get_pos() or 0 # Initialize gripper position
else:
    current_pos = 0

print("""6-DOF Robot Control Using Keyboard. Press 'q' to quit.
TRANSLATION=> a: Left, d: Right, w: Forward, s: Backward, z: Up, x: Down
ROTATION=> j: Left Upward, l: Right Upward,  i: Forward, k: Backward, n: Rotate left, m: Rotate right
GRIPPER=> left arrow: close, right arrow: open""")

# Initiate jogging with 0 speed
rtde_control.jogStart(speed_vector, RTDEControlInterface.FEATURE_TOOL)

# Keyboard control
try:
    while True:
        t_start = rtde_control.initPeriod() # To time each cycle

        if keyboard.is_pressed("q"): # Quit
            break

        # Translation
        elif keyboard.is_pressed("a"): # Left
            speed_vector[0] = -linear_speed_magnitude
        elif keyboard.is_pressed("d"): # Right
            speed_vector[0] = linear_speed_magnitude
        elif keyboard.is_pressed("w"): # Forward
            speed_vector[1] = -linear_speed_magnitude
        elif keyboard.is_pressed("s"): # Backward
            speed_vector[1] = linear_speed_magnitude
        elif keyboard.is_pressed("z"): # Up
            speed_vector[2] = -linear_speed_magnitude
        elif keyboard.is_pressed("x"): # Down
            speed_vector[2] = linear_speed_magnitude

        # Rotation
        elif keyboard.is_pressed("i"): # Forward
            speed_vector[3] = angular_speed_magnitude
        elif keyboard.is_pressed("k"): # Backward
            speed_vector[3] = -angular_speed_magnitude
        elif keyboard.is_pressed("j"): # Left Upward
            speed_vector[4] = -angular_speed_magnitude
        elif keyboard.is_pressed("l"): # Right Upward
            speed_vector[4] = angular_speed_magnitude
        elif keyboard.is_pressed("n"): # Rotate left
            speed_vector[5] = angular_speed_magnitude
        elif keyboard.is_pressed("m"): # Rotate right
            speed_vector[5] = -angular_speed_magnitude

        # Gripper control
        elif gripper_enabled and keyboard.is_pressed("left"): # Close
            current_pos = min(255, current_pos + i)
            gripper.move(current_pos)

        elif gripper_enabled and keyboard.is_pressed("right"): # Open
            current_pos = min(255, current_pos - i)
            gripper.move(current_pos)

        # No movement if no key is pressed
        else:
            speed_vector = [0.0] * 6

        rtde_control.jogStart(speed_vector, RTDEControlInterface.FEATURE_TOOL) # Implement movement for key press
        rtde_control.waitPeriod(t_start) # To maintain constant cycle duration

except KeyboardInterrupt:
    pass
finally:
    rtde_control.jogStop()
    rtde_control.stopScript()
    if gripper and gripper_enabled:
        gripper.close()
    print("Jogging stopped.")