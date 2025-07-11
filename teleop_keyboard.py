"""Python code to control UR 6-DOF TCP movement and gripper opening/closing with a keyboard"""
"""Run code as sudo"""
"""Activate gripper prior to start"""

import keyboard
from rtde_control import RTDEControlInterface
from robotiq_socket_gripper import RobotiqSocketGripper

# Connect to robot and gripper
robot_ip = "192.168.1.223" # Ensure static IP addresses of robot and computer match
rtde_control = RTDEControlInterface(robot_ip)

gripper = RobotiqSocketGripper(robot_ip)
gripper.connect()

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
gripper.set_force(100) # Range: 0 (min) to 255 (max)
gripper.set_speed(50) # Range: 0 (min) to 255 (max)
current_pos = gripper.get_pos() or 0 # Initialize gripper position

print("""6-DOF Robot Control. Press 'q' to quit.
TRANSLATION=> a: Left, d: Right, w: Up, s: Down, z: Forward, x: Backward
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
            speed_vector = [linear_speed_magnitude, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif keyboard.is_pressed("d"): # Right
            speed_vector = [-linear_speed_magnitude, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif keyboard.is_pressed("w"): # Up
            speed_vector = [0.0, 0.0, -linear_speed_magnitude, 0.0, 0.0, 0.0]
        elif keyboard.is_pressed("s"): # Down
            speed_vector = [0.0, 0.0, linear_speed_magnitude, 0.0, 0.0, 0.0]
        elif keyboard.is_pressed("y"): # Forward (y for German keyboard, change to z if required)
            speed_vector = [0.0, -linear_speed_magnitude, 0.0, 0.0, 0.0, 0.0]
        elif keyboard.is_pressed("x"): # Backward
            speed_vector = [0.0, linear_speed_magnitude, 0.0, 0.0, 0.0, 0.0]

        # Rotation
        elif keyboard.is_pressed("i"): # Forward
            speed_vector = [0.0, 0.0, 0.0, angular_speed_magnitude, 0.0, 0.0]
        elif keyboard.is_pressed("k"): # Backward
            speed_vector = [0.0, 0.0, 0.0, -angular_speed_magnitude, 0.0, 0.0]
        elif keyboard.is_pressed("j"): # Left Upward
            speed_vector = [0.0, 0.0, 0.0, 0.0, angular_speed_magnitude, 0.0]
        elif keyboard.is_pressed("l"): # Right Upward
            speed_vector = [0.0, 0.0, 0.0, 0.0, -angular_speed_magnitude, 0.0]
        elif keyboard.is_pressed("n"): # Rotate left
            speed_vector = [0.0, 0.0, 0.0, 0.0, 0.0, angular_speed_magnitude]
        elif keyboard.is_pressed("m"): # Rotate right
            speed_vector = [0.0, 0.0, 0.0, 0.0, 0.0, -angular_speed_magnitude]

        # Gripper control
        elif keyboard.is_pressed("left"): # Close
            current_pos = min(255, current_pos + i)
            gripper.move(current_pos)

        elif keyboard.is_pressed("right"): # Open
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
    print("Jogging stopped.")
