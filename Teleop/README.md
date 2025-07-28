# ⚙️ UR3e Teleoperation
This folder contains two methods to teleoperate a UR3e cobot with a wrist-attached Robotiq 2F-140 two-finger gripper using Python: one using a keyboard, and the other using a PlayStation-5 (DualSense) controller.

The functioning of each method is described below.

## Keyboard-Based Teleoperation
6-DOF UR3e keyboard-based teleoperation can be done using the file '**teleop_keyboard.py**'.

### Key features
This code allows for:

1. **3-DOF translation using the keys:**

    *WSAD:* Forward / backward, Left / right

    *ZX:* Up / down


2.  **3-DOF rotation using the keys:**

    *IKJL:* Rotate forward / backward, Left upward / right upward

    *NM:* Rotate left / right about vertical axis


3. **Robotiq 2F-140 gripepr control using the keys:**

    *Left arrow:* Gripper close

    *Right arrow:* Gripper open

### Requirements
This code must be run as sudo due to the use of the 'keyboard' library.

## Class RobotiqSocketGripper
In both methods of teleoperation, Robotiq gripper control is achieved using the class RobotiqSocketGripper. 
