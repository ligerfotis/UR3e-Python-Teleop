"""Python code to control UR 6-DOF TCP pose with a PlayStation controller using RTDE servoL.
Tested with a PS5 DualSense controller and a UR3e cobot.
Gripper optional: script falls back to robot-only control if absent."""

import time

import pygame
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from robotiq_socket_gripper import RobotiqSocketGripper


# Connect to robot and gripper
robot_ip = "192.168.1.201"  # Ensure static IP addresses of robot and computer match
rtde_control = RTDEControlInterface(robot_ip)
rtde_receive = RTDEReceiveInterface(robot_ip)

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

Pose vector ->
[x, y, z, rx, ry, rz] (TCP pose in base frame, UR axis-angle representation)
"""

# Speed parameters (max velocities)
linear_speed_magnitude = 0.05  # m/s (translational max speed)
angular_speed_magnitude = 0.25  # rad/s (rotational max speed)

# ServoL parameters
cycle_time = 0.01  # [s] control period ~100 Hz
servo_a = 0.5      # tool acceleration [m/s^2]
servo_v = 0.25     # tool speed [m/s]
servo_lookahead_time = 0.1
servo_gain = 300

# Gripper parameters
i = 1  # Increment for gripper movement
if gripper_enabled:
    gripper.set_force(20)  # Range: 0 (min) to 255 (max)
    gripper.set_speed(50)  # Range: 0 (min) to 255 (max)
    current_pos = gripper.get_pos() or 0  # Initialize gripper position
else:
    current_pos = 0

print(
    """6-DOF Robot TCP Servo Control Using PS5 Controller (servoL). Press 'Option' button (three horizontal lines, right of controller) to quit.
LEFT JOYSTICK: Translate right/left, forward/backward
RIGHT JOYSTICK: Rotate about x-axis, Rotate about y-axis
L1, L2: Translate up/down
R1, R2: Rotate about Z-axis (vertical axis)
SQUARE: Close gripper
CIRCLE: Open gripper"""
)


# Initialize PlayStation controller
pygame.joystick.init()  # Initialize joystick control
pygame.display.init()  # Initialize event handling
joy = pygame.joystick.Joystick(0)  # Detect one controller
joy.init()  # Initialize controller


# Function to rescale joystick axis position according to initial resting position
def rescale_axis(axis_position, axis_ini_position):
    """
    Parameters:
        axis_position: Axis position of joystick (value between -1 and 1)
        axis_ini_position: Axis position of joystick at rest (value between -1 and 1)

    Returns:
        rescaled_position: Rescaled axis position of joystick (value between -1 and 1)
    """
    low_segment_len = 1 + axis_ini_position  # Distance from initial position to minimum possible position (-1)
    upper_segment_len = 1 - axis_ini_position  # Distance from initial position to maximum possible position (-1)

    if axis_position < axis_ini_position:  # If joystick is moved in negative relative direction
        rescaled_position = (axis_position - axis_ini_position) / low_segment_len  # Normalize
    else:  # If joystick is moved in positive relative direction
        rescaled_position = (axis_position - axis_ini_position) / upper_segment_len  # Normalize

    return rescaled_position


# Function to prevent drifting when joysticks are not being controlled
def apply_deadzone(value, deadzone=0.1):  # Default deadzone=0.1 from [-1, 1]
    if abs(value) < deadzone:
        return 0.0
    return value


# Update inputs once before reading initial axes
pygame.event.pump()

# Get initial resting positions
axis0_ini = joy.get_axis(0)  # Left stick left/right
axis1_ini = joy.get_axis(1)  # Left stick up/down
axis2_ini = joy.get_axis(3)  # Right stick left/right
axis3_ini = joy.get_axis(4)  # Right stick up/down

# Get initial TCP pose as starting target
target_pose = list(rtde_receive.getActualTCPPose())

try:
    while True:
        loop_start = time.time()

        if joy.get_button(9):  # Quit (Options button)
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

        # Compute desired TCP velocities in tool frame approximation
        vx = linear_speed_magnitude * axis0  # Left/right
        vy = linear_speed_magnitude * axis1  # Forward/Backward

        if joy.get_button(6):  # L1 (up)
            vz = linear_speed_magnitude
        elif joy.get_button(4):  # L2 (down)
            vz = -linear_speed_magnitude
        else:
            vz = 0.0

        wx = -angular_speed_magnitude * axis3  # Rotate up/down
        wy = angular_speed_magnitude * axis2  # Rotate left/right

        if joy.get_button(7):  # R1 (Rotate left about Z-axis)
            wz = angular_speed_magnitude
        elif joy.get_button(5):  # R2 (Rotate right about Z-axis)
            wz = -angular_speed_magnitude
        else:
            wz = 0.0

        # Integrate velocities into target pose (small increments each cycle)
        target_pose[0] += vx * cycle_time
        target_pose[1] += vy * cycle_time
        target_pose[2] += vz * cycle_time
        target_pose[3] += wx * cycle_time
        target_pose[4] += wy * cycle_time
        target_pose[5] += wz * cycle_time

        # Simple debug print when there is a non-zero command
        if abs(vx) + abs(vy) + abs(vz) + abs(wx) + abs(wy) + abs(wz) > 1e-3:
            print(f"cmd v=({vx:.3f},{vy:.3f},{vz:.3f}) w=({wx:.3f},{wy:.3f},{wz:.3f}) pose={target_pose}")

        # Gripper control
        if gripper_enabled and joy.get_button(3):  # Square (Close gripper)
            current_pos = min(255, current_pos + i)
            gripper.move(current_pos)

        elif gripper_enabled and joy.get_button(1):  # Circle (Open gripper)
            current_pos = max(0, current_pos - i)
            gripper.move(current_pos)

        # Send servoL command
        rtde_control.servoL(
            target_pose,
            servo_a,
            servo_v,
            cycle_time,
            servo_lookahead_time,
            servo_gain,
        )

        # Maintain roughly constant loop period
        elapsed = time.time() - loop_start
        sleep_time = cycle_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    pass
finally:
    # Stop robot program
    rtde_control.stopScript()
    if gripper and gripper_enabled:
        gripper.close()
    print("TCP servoL control stopped.")


