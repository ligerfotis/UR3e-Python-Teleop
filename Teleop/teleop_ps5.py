"""Python code to control UR 6-DOF TCP movement and gripper opening/closing with a PlayStation controller"""
""""Tested using a PS5 DualSense Controller and a UR3e cobot"""
"""Activate gripper prior to start"""

import pygame
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

print("""6-DOF Robot Control. Press 'Option' button to quit.
LEFT JOYSTICK: Translate right/left, forward/backward
RIGHT JOYSTICK: Rotate left/right, up/down
L1, L2: Up/down
R1, R2: Rotate left/right about Z-axis
Square: Close gripper
Circle: Open gripper""")

# Initiate jogging with 0 speed
rtde_control.jogStart(speed_vector, RTDEControlInterface.FEATURE_TOOL)

# Initialize PlayStation controller
pygame.joystick.init() # Initialize joystick control
pygame.display.init() # Initialize event handling
joy = pygame.joystick.Joystick(0) # Detect one controller
joy.init() # Initialize controller


# Function to rescale joystick axis position according to initial resting position
def rescale_axis(axis_position, axis_ini_position):
    """
    Parameters:
        axis_position: Axis position of joystick (value between -1 and 1)
        axis_ini_position: Axis position of joystick at rest (value between -1 and 1)

    Returns:
        rescaled_position: Rescaled axis position of joystick (value between -1 and 1)
    """
    low_segment_len = 1 + axis_ini_position # Distance from initial position to minimum possible position (-1)
    upper_segment_len = 1 - axis_ini_position # Distance from initial position to maximum possible position (-1)

    if axis_position < axis_ini_position: # If joystick is moved in negative relative direction
        rescaled_position = (axis_position - axis_ini_position) / low_segment_len # Normalize
    else: # If joystick is moved in positive relative direction
        rescaled_position = (axis_position - axis_ini_position) / upper_segment_len # Normalize

    return rescaled_position


# Function to prevent drifting when joysticks are not being controlled
def apply_deadzone(value, deadzone=0.1): # Default deadzone=0.1 from [-1, 1]
    if abs(value) < deadzone:
        return 0.0
    return value


# Update inputs
pygame.event.pump()

# Get initial resting positions
axis0_ini = joy.get_axis(0) # Left stick left/right
axis1_ini = joy.get_axis(1) # Left stick up/down
axis2_ini = joy.get_axis(3) # Right stick left/right
axis3_ini = joy.get_axis(4) # Right stick up/down

try:
    while True:
        t_start = rtde_control.initPeriod()  # To time each cycle

        if joy.get_button(9): # Quit
            break

        # Update inputs
        pygame.event.pump()

        # Rescale axes
        axis0 = rescale_axis(joy.get_axis(0), axis0_ini)
        axis1 = rescale_axis(joy.get_axis(1), axis1_ini)
        axis2 = rescale_axis(joy.get_axis(3), axis2_ini)
        axis3 = rescale_axis(joy.get_axis(4), axis3_ini)

        # Apply deadzone to prevent drifting
        axis0 = apply_deadzone(axis0)
        axis1 = apply_deadzone(axis1)
        axis2 = apply_deadzone(axis2)
        axis3 = apply_deadzone(axis3)

        # Translation
        # Left joystick
        speed_vector[0] = linear_speed_magnitude * axis0  # Left/right
        speed_vector[1] = linear_speed_magnitude * axis1  # Forward/Backward

        if joy.get_button(6): # L1 (up)
            speed_vector[2] = linear_speed_magnitude
        elif joy.get_button(4): # L2 (down)
            speed_vector[2] = -linear_speed_magnitude
        else:
            speed_vector[2] = 0.0

        # Rotation
        # Right joystick
        speed_vector[3] = -angular_speed_magnitude * axis3  # Rotate up/down
        speed_vector[4] = angular_speed_magnitude * axis2  # Rotate left/right

        if joy.get_button(7): # R1 (Rotate left about Z-axis)
            speed_vector[5] = angular_speed_magnitude
        elif joy.get_button(5): # R2 (Rotate right about Z-axis)
            speed_vector[5] = -angular_speed_magnitude
        else:
            speed_vector[5] = 0.0

        # Gripper control
        if joy.get_button(3): # Square (Close gripper)
            current_pos = min(255, current_pos + i)
            gripper.move(current_pos)

        elif joy.get_button(1): # Circle (Open gripper)
            current_pos = min(255, current_pos - i)
            gripper.move(current_pos)

        rtde_control.jogStart(speed_vector, RTDEControlInterface.FEATURE_TOOL) # Implement movement for PS5 input
        rtde_control.waitPeriod(t_start) # To maintain constant cycle duration

except KeyboardInterrupt:
    pass
finally:
    rtde_control.jogStop()
    rtde_control.stopScript()
    print("Jogging stopped.")
